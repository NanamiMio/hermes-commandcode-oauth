"""Selected CLI 1.87.1 registry facts, verified 2026-10-22.

No network or installed CLI dependency on Hermes' per-request capability path.
Unknown models remain undeclared instead of being assumed vision-capable.
"""

MODEL_METADATA = {'deepseek/deepseek-v4-pro': {'context_window': 1000000,
                              'supports_vision': False,
                              'supports_tools': True,
                              'supports_reasoning': True,
                              'efforts': ['off', 'high', 'max']},
 'deepseek/deepseek-v4-flash': {'context_window': 1000000,
                                'supports_vision': False,
                                'supports_tools': True,
                                'supports_reasoning': True,
                                'efforts': ['off', 'high', 'max']},
 'deepseek/deepseek-v4.1-flash': {'context_window': 1000000,
                                  'supports_vision': True,
                                  'supports_tools': True,
                                  'supports_reasoning': True,
                                  'efforts': ['off', 'low', 'high', 'max']},
 'moonshotai/Kimi-K3': {'context_window': 1000000,
                        'supports_vision': True,
                        'supports_tools': True,
                        'supports_reasoning': True,
                        'efforts': ['low', 'high', 'max']},
 'z-ai/glm-5.3-flash': {'context_window': 1048576,
                        'supports_vision': True,
                        'supports_tools': True,
                        'supports_reasoning': True,
                        'efforts': ['low', 'high', 'max']},
 'minimax/minimax-m3.1-flash': {'context_window': 1000000,
                                'supports_vision': True,
                                'supports_tools': True,
                                'supports_reasoning': True,
                                'efforts': ['low', 'medium', 'high', 'xhigh', 'max']},
 'minimax/minimax-m3.1-flash:free': {'context_window': 1000000,
                                     'supports_vision': True,
                                     'supports_tools': True,
                                     'supports_reasoning': True,
                                     'efforts': ['low', 'medium', 'high', 'xhigh', 'max']},
 'MiniMaxAI/MiniMax-M3': {'context_window': 1000000,
                          'supports_vision': True,
                          'supports_tools': True,
                          'supports_reasoning': True,
                          'efforts': ['low', 'medium', 'high']},
 'Qwen/Qwen3.8-Omni-Flash': {'context_window': 1000000,
                             'supports_vision': True,
                             'supports_tools': True,
                             'supports_reasoning': True,
                             'efforts': ['low', 'medium', 'xhigh']},
 'Qwen/Qwen3.7-Max': {'context_window': 1000000,
                      'supports_vision': False,
                      'supports_tools': True,
                      'supports_reasoning': False,
                      'efforts': []},
 'poolside/laguna-s-2.1-free': {'context_window': 256000,
                                'supports_vision': False,
                                'supports_tools': True,
                                'supports_reasoning': False,
                                'efforts': []},
 'meta/muse-spark-1.3': {'context_window': 1048576,
                         'supports_vision': True,
                         'supports_tools': True,
                         'supports_reasoning': True,
                         'efforts': ['low', 'medium', 'high', 'xhigh', 'max']},
 'meta/muse-spark-1.3-contributor': {'context_window': 1048576,
                                     'supports_vision': True,
                                     'supports_tools': True,
                                     'supports_reasoning': True,
                                     'efforts': ['low', 'medium', 'high', 'xhigh']}}

def model_info(model):
    return next((info for key, info in MODEL_METADATA.items() if key.lower() == (model or "").lower()), None)


def capability_overrides():
    result = {}
    for model, info in MODEL_METADATA.items():
        caps = {key: value for key, value in info.items() if key != "efforts"}
        result[model] = caps
        result[model.lower()] = caps
    return result
