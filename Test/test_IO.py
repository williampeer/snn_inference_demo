from IO import *
from Models.LIF import LIF


def test_save_model():
    model = LIF(device='cpu', parameters={})
    save(model, loss=False, uuid='test_uuid')
    save_model_params(model, 'test_uuid')
    save_entire_model(model, 'test_uuid')


test_save_model()
