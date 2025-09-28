class ModulatedPreTrainedModel:
    def __init__(self):
        self.modulator = ...

    def modulate_base_model(self, ids):
        self.modulator.apply(self.base_model, ids)

    def forward(self):
        with self.modulate_base_model(ids):
            self.base_model()

    def generate(self):
        with self.modulate_base_model(ids):
            self.base_model.generate(...)
