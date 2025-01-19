This is the notebook where the model for PianoAI was trained, and the code for generating it and returning it through an API currently hosted on an EC2.

The dataset used is the Maestro dataset, a collection of MIDI data of piano compositions, from which I took note-on events and created regression-like data of 
the pitch (as a MIDI number) of the each note as well as how many 16th notes (uration) were in between the current and last note. The model used is a 
simple Bidirectional LSTM-based model that projects the duration and pitch into a higher encoding dimension and uses an LSTM-based stack to predict the next 
note and it's duration, similarly to how a transformer predicts. Other approaches like attention and classifiying pitch and duration data into individual classes 
were tried as well, however the simpler regression and model derived the best results.

The model achieved results that show its ability to recognize patterns and create variety since it was traiend with both factors in mind, however its generation has room for improvement. 
One method I plan on looking in to is a dataset that simplifies note generation with a raw note sequence dataset instead of taking it through MIDI data, since it doesn't accurately 
represnet all aspects of notes and also makes the task harder to interpret. I also plan on refining the architecture and researching different training techniques used 
by professional music models, such as Google's Magenta, to create a more intuitive and successful model.
