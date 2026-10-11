// Large prompt attention tiles over prepared compact Q and persistent KV.
// Device tensor operands avoid restaging Q/K/V through threadgroup memory.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
using PrefillQKV = tensor<device bfloat, dextents<int, 2>, tensor_inline>;
using PrefillScore = tensor<threadgroup float, dextents<int, 2>, tensor_inline>;
using PrefillProb = tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline>;
// KS: the softmax/value block. Scores come from one QM x KN product; each KS-key block keeps its own maximum,
// rescale and value product, in key order, so KS = 32 reproduces 32-key tiles bit for bit with a quarter of the
// barriers at KN = 128.
#ifndef KS
#define KS KN
#endif
#ifndef SCORE_BF16
#define SCORE_BF16 0                 // 1: the scores' first step is BF16 rounding, so store them as BF16 and overwrite them
#endif                               // with the probabilities in place: a third of the threadgroup memory, same values
constexpr constant auto prefill_score = matmul2d_descriptor(QM, KN, D, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
constexpr constant auto prefill_value = matmul2d_descriptor(QM, D, KS, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);

kernel void gqa_decode_mma(device bfloat* q [[buffer(0)]], device bfloat* k [[buffer(1)]], device bfloat* v [[buffer(2)]],
    device float* part_o [[buffer(7)]], device float* part_md [[buffer(8)]], constant GqaParams& p [[buffer(9)]],
    device const StepState* st [[buffer(15)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
#if SCORE_BF16
  threadgroup float carry[2*QM*(KN/KS)], md[2*QM];
#else
  threadgroup float score[QM*KN], carry[2*QM*(KN/KS)], md[2*QM];
#endif
  threadgroup bfloat prob[QM*KN];
  if(st->done || st->t_this_step==0u) return;
  const uint T=st->t_this_step,pos=st->position,ctx=pos+T;
  const uint rep=p.heads/p.kv_heads, rows=T*rep, chunks=(ctx+CH-1u)/CH, groups=(rows+QM-1u)/QM;
  matmul2d<prefill_score,execution_simdgroups<MMA_SG>> score_op;
  matmul2d<prefill_value,execution_simdgroups<MMA_SG>> value_op;
  PrefillQKV kt(k,dextents<int,2>(p.kv_heads*D,p.ctx_max));
  PrefillQKV vt(v,dextents<int,2>(p.kv_heads*D,p.ctx_max));
  PrefillProb pt(prob,dextents<int,2>(KN,QM));
#if !SCORE_BF16
  PrefillScore sc(score,dextents<int,2>(KN,QM));
#endif
  for(uint task=group;task<p.kv_heads*chunks*groups;task+=p.n_sg) {
    const uint h=task/(chunks*groups),chunk=(task/groups)%chunks,r0=(task%groups)*QM;
    PrefillQKV qt(q+(ulong)h*p.rows_max*D,dextents<int,2>(D,p.rows_max));
    auto a=qt.slice<D,QM>(0,r0);
    auto b=kt.slice<D,KN>(h*D,chunk*CH);
    auto vv=vt.slice<D,KS>(h*D,chunk*CH);
    auto ps=pt.slice<KS,QM>(0,0);
    auto accum=value_op.get_destination_cooperative_tensor<decltype(ps),decltype(vv),float>();
    for(uint16_t i=0;i<accum.get_capacity();i++) if(accum.is_valid_element(i)) accum[i]=0;
    for(uint r=sg;r<QM;r+=MMA_SG) if(lane==0) {md[2*r]=-INFINITY;md[2*r+1]=0;}
    const uint key_end=min(ctx,pos+(r0+QM-1u)/rep+1u),key_stop=min(key_end,(chunk+1)*CH);
    for(uint key0=chunk*CH;key0<key_stop;key0+=KN) {
      b=kt.slice<D,KN>(h*D,key0);
      auto scores=score_op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
      for(uint16_t i=0;i<scores.get_capacity();i++) if(scores.is_valid_element(i)) scores[i]=0;
      score_op.run(a,b,scores);
#if SCORE_BF16
      for(uint16_t i=0;i<scores.get_capacity();i++) if(scores.is_valid_element(i)) {
        auto x=scores.get_multidimensional_index(i);
        prob[x[1]*KN+x[0]]=bfloat(scores[i]);                 // round to nearest even, as round_bf16
      }
#else
      scores.store(sc);
#endif
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint r=sg;r<QM;r+=MMA_SG) {
        const uint row=r0+r,t=row/rep;
        for(uint s=0;s<KN/KS && key0+s*KS<key_stop;s++) {
          float values[KS/32],m=-INFINITY;
          for(uint u=0;u<KS/32;u++) {
            const uint key=key0+s*KS+lane+32*u;
#if SCORE_BF16
            values[u]=row<rows && key<ctx && key<=pos+t ? round_bf16(float(prob[r*KN+s*KS+lane+32*u])*p.scaling) : -INFINITY;
#else
            values[u]=row<rows && key<ctx && key<=pos+t ? round_bf16(round_bf16(score[r*KN+s*KS+lane+32*u])*p.scaling) : -INFINITY;
#endif
            m=max(m,values[u]);
          }
          m=simd_max(m);float den=0;
          for(uint u=0;u<KS/32;u++) {
            float pr=values[u]==-INFINITY ? 0.0f : exp(values[u]-m);
            prob[r*KN+s*KS+lane+32*u]=bfloat(pr);den+=pr;
          }
          den=simd_sum(den);
          if(lane==0) {
            const float old_m=md[2*r],new_m=max(old_m,m);
            const float a=old_m==-INFINITY ? 0.0f : exp(old_m-new_m);
            const float b=m==-INFINITY ? 0.0f : exp(m-new_m);
            md[2*r]=new_m;md[2*r+1]=a*md[2*r+1]+b*den;
            carry[2*(s*QM+r)]=a;carry[2*(s*QM+r)+1]=b;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint s=0;s<KN/KS && key0+s*KS<key_stop;s++) {
        vv=vt.slice<D,KS>(h*D,key0+s*KS);
        ps=pt.slice<KS,QM>(s*KS,0);
        auto out=value_op.get_destination_cooperative_tensor<decltype(ps),decltype(vv),float>();
        for(uint16_t i=0;i<out.get_capacity();i++) if(out.is_valid_element(i)) out[i]=0;
        value_op.run(ps,vv,out);
        for(uint16_t i=0;i<out.get_capacity();i++) if(out.is_valid_element(i)) {
          const uint r=out.get_multidimensional_index(i)[1];
          accum[i]=carry[2*(s*QM+r)]*accum[i]+carry[2*(s*QM+r)+1]*out[i];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint16_t i=0;i<accum.get_capacity();i++) if(accum.is_valid_element(i)) {
      auto x=accum.get_multidimensional_index(i);const uint row=r0+x[1];
      if(row<rows) part_o[((h*p.n_chunks_max+chunk)*p.rows_max+row)*D+x[0]]=accum[i];
    }
    for(uint r=sg;r<QM;r+=MMA_SG) if(lane==0 && r0+r<rows) {
      const uint base=((h*p.n_chunks_max+chunk)*p.rows_max+r0+r)*2;
      part_md[base]=md[2*r];part_md[base+1]=md[2*r+1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
