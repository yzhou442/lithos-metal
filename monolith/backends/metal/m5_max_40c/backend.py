from ..base import MetalBackend
from .scheduling import finalize
from .attention import direct_attention_shape


class Backend(MetalBackend):
    """Measured 40-core Max fusion policy and explicit draft/decoder recipes."""
    id = "m5_max_40c"
    finalize = staticmethod(finalize)
    direct_attention_shape = staticmethod(direct_attention_shape)

    def optimize_prefill(self, program):
        from .prefill import optimize
        return optimize(program)

    def validate_config(self, config):
        from .validation import validate_gdn_config
        validate_gdn_config(config.name, config.gdn_mixer_fusion, config.gpu_cores)

    def serving_recipes(self, model, drafter, quantization):
        from .serving import recipes
        return recipes(model, drafter, quantization)

    def serving_prefill_chunk(self, model, drafter, recipes):
        # prefill.py tunes 512-row Qwen3.8-27B projection/attention tiles; the
        # serving recipes match exactly that model (and its DSpark head).
        if recipes and any(r.get('target') for r in recipes.values() if isinstance(r, dict)):
            return 512
        return None

    def serving_verify_costs(self, recipes):
        # measured round costs of the recipes' workload only (serving.verify_costs)
        from .serving import verify_costs
        return verify_costs(recipes)
