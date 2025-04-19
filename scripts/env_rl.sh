pip install -e ".[train]"

pip install vllm==0.7.2

pip install wheel
pip install flash-attn --no-build-isolation

cd /openseg_blob/wxc/trl-latest
pip install -e .


pip uninstall bitsandbytes -y
pip install outlines==0.0.46
pip install latex2sympy2_extended math_verify

pip install clint

apt-get install python3-tk -y