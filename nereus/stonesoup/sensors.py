class SonarSensor(Sensor):
    def __init__(self, propagation_model, signal_model, **kwargs):
        super().__init__(**kwargs)
        self.propagation_model = propagation_model
        self.signal_model = signal_model
