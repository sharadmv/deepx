import os, json

CONFIG = {
    'backend': 'tensorflow',
    'epsilon': 1e-7,
    'floatx': 'float32'
}

deepx_dir = os.path.expanduser(os.path.join('~', '.deepx'))
if not os.path.exists(deepx_dir):
    os.makedirs(deepx_dir)

config_path = os.path.join(deepx_dir, 'deepx.json')
if os.path.exists(config_path):
    with open(config_path, 'r') as fp:
        config = json.load(fp)
    CONFIG.update(config)
else:
    with open(config_path, 'w') as fp:
        json.dump(CONFIG, fp)

CONFIG['backend'] = os.environ.get("DEEPX_BACKEND", CONFIG["backend"])

def set_backend(backend_name):
    CONFIG['backend'] = backend_name

def get_backend():
    return CONFIG['backend']
