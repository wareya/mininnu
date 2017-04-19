# mininnu
Mini (code-wise) neural network upscaler

Structure: Loads two files. One is for training and the other is upscaling.

Training file must already be lowpassed for performance/simplicity's sake.

This demonstration is interesting because it's not inherently bound to neural networks or one particular kind of neural network. Any system that can act like a trainable data structure, even a lookup table, can be fitted around mininnu.

Mininnu also shows the normal way to normalize waveform-like inputs to neural networks.

Mininnu is also a proof that recursive crosshatch upscalers can work properly if the upscaling function itself is clean enough.

mininnu.cpp is released to the public domain and under the ISC license.
