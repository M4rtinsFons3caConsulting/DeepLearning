README.md

Our best model was trained on a previous iteration of the notebook, as to our own astonishement we were unable to improve beyond our first preprocessing, and augmentation schema and our simple yet effective approach with EfficientNetB4. 

We therefore tailored the refactored code for our base_model - `base_model_00`-  notebooks to load the respective weights and arquitecture of the model load them and our focal loss and weights custom constraints and made predictions on our testgenerator items. As our process is entirely controled vida the configurations files, we can be certain that the results are repreducible, given our use of constant random state.

Therefore, to assess the expected flow of code, refer to the base_model present in the `family/` directory whereas, for the results refer to both files present in this directory, first showing the top one loading the weights and the bottom one showcasing the complete run, that was halted by a botched metric call.