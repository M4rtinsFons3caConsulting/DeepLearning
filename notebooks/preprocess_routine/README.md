To ensure our samples can be used to effectivelly train a model, and after our initial
inspection of the original directories contents, we became convinced that some level of 
general image preprocessing was in order.

To that end, a sequence of steps was idealized. First the images would be cleaned, as 
some of them required were in dire need of such treament, then oversampling would be
performed with "difficult enough" transformations, that would improve generalization
rather than being simply copies of the originals. And all resulting images would be
resized, so that they would be ready to flow into a trainable model pipeline.

The scripts that accomplish these steps are mostly located inside of the deep package,
in the preprocessing sub-module and they are run here in these notebooks.
