from stream_voice_focus import devices as dv


def D(name, i=0, o=0, idx=0):
    return dv.DeviceInfo(idx, name, dv.WASAPI, i, o, 48000.0)


CABLE_OUT = D("CABLE Output (VB-Audio Virtual Cable)", i=2, idx=1)
CABLE_IN = D("CABLE Input (VB-Audio Virtual Cable)", o=2, idx=2)
CABLE_IN16 = D("CABLE In 16ch (VB-Audio Virtual Cable)", o=2, idx=3)
HP = D("Headphones (oraimo BoomPop Pro)", o=2, idx=4)
MIC = D("Microphone (Logi C270 HD WebCam)", i=2, idx=5)
VM_IN = D("Voicemeeter Input (VB-Audio Voicemeeter VAIO)", o=2, idx=6)


def levels(issues):
    return [i.level for i in issues]


def test_same_cable_is_error():
    assert "error" in levels(dv.check_routing(CABLE_OUT, CABLE_IN))
    assert "error" in levels(dv.check_routing(CABLE_OUT, CABLE_IN16))


def test_normal_route_is_clean():
    assert dv.check_routing(CABLE_OUT, HP) == []


def test_microphone_input_warns_and_virtual_output_warns():
    assert "warning" in levels(dv.check_routing(MIC, HP))
    assert "warning" in levels(dv.check_routing(CABLE_OUT, VM_IN))


def test_missing_selection_is_error():
    assert levels(dv.check_routing(None, HP)) == ["error"]


def test_defaults_never_pick_microphone_or_cable_output():
    devs = [CABLE_OUT, CABLE_IN, HP, MIC, VM_IN]
    assert dv.default_input_device(devs) is CABLE_OUT
    out = dv.default_output_device(devs)
    assert out is not None and not dv.is_virtual_playback(out.name)
    assert dv.default_input_device([MIC, HP]) is None


def test_find_vb_cable_and_ident():
    cap, play = dv.find_vb_cable([CABLE_OUT, CABLE_IN, HP])
    assert cap is CABLE_OUT and play is CABLE_IN
    assert dv.find_by_ident(HP.ident, [CABLE_OUT, HP]) is HP
    assert dv.find_vb_cable([HP, MIC]) == (None, None)
    assert "vb-audio.com" in dv.vb_cable_missing_message()


def test_real_enumeration_runs():
    devs = dv.list_devices()
    assert isinstance(devs, list)
