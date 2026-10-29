"""Tests the complete action boundary, not frozen trajectory reproduction."""
import copy
import numpy as np
import pytest
from capstone_vla.ball_action_contract import (ACTION_DIM, STATE_DIM, SLICES, FIXED_INPUTS,
    VARIABLE_INPUTS, decode_action, encode_packet, encode_stream, decode_stream,
    model_configuration, robot_state, audit_contract)


def sample():
    action=np.zeros(ACTION_DIM,dtype=np.float32)
    action[SLICES['target_position']]=[1.,2.,3.]
    action[SLICES['target_quaternion']]=[1.,0.,0.,0.]
    return decode_action(action,0,[0.]*7)


def test_all_native_fields_classified_and_float32_roundtrip():
    packet=sample();packet['inputs'].update(target_linear_velocity=[.1,-.2,.3],
        target_linear_acceleration=[1.,2.,3.],linear_velocity_reference=[.2,.3,.4],
        maximum_relative_linear_speed_mps=.5,orientation_axis_world=[0.,0.,1.],
        orientation_roll_weight=.3,joint_limit_margin_rad=.08,
        preserve_orientation_during_translation=True,operational_orientation_task=True)
    packet['gripper_closed']=True
    action=encode_packet(packet);decoded=decode_action(action,0,[0.]*7)
    assert action.shape==(29,) and action.dtype==np.float32
    assert set(decoded['inputs'])==set(FIXED_INPUTS)|VARIABLE_INPUTS|{'posture_target'}
    for key in VARIABLE_INPUTS:
        np.testing.assert_allclose(decoded['inputs'][key],packet['inputs'][key],atol=1.e-7)
    assert decoded['gripper_closed'] is True
    assert all(decoded['safety_callbacks'].values())


def test_optional_zero_and_absent_are_distinct():
    packet=sample();packet['inputs']['target_linear_acceleration']=[0.]*3
    present=encode_packet(packet)
    packet['inputs']['target_linear_acceleration']=None
    absent=encode_packet(packet)
    assert present[SLICES['acceleration_enabled']]==1
    assert absent[SLICES['acceleration_enabled']]==0
    assert decode_action(present,0)['inputs']['target_linear_acceleration']==[0.]*3
    assert decode_action(absent,0)['inputs']['target_linear_acceleration'] is None


def test_binary_gripper_event_survives_each_control_tick():
    packets=[]
    for i,closed in enumerate([False,True,False,True]):
        p=sample();p.update(frame=i,gripper_closed=closed);packets.append(p)
    decoded=decode_stream(encode_stream(packets))
    assert [p['gripper_closed'] for p in decoded]==[False,True,False,True]


def test_quaternion_sign_continuity_and_decoder_normalization():
    p=sample();other=copy.deepcopy(p);other['frame']=1
    other['inputs']['target_quaternion']=[-1.,0.,0.,0.]
    actions=encode_stream([p,other])
    np.testing.assert_array_equal(actions[0],actions[1])
    actions[1,SLICES['target_quaternion']]*=2
    assert decode_stream(actions)[1]['inputs']['target_quaternion']==[1.,0.,0.,0.]


def test_unknown_or_changed_fixed_field_rejected_instead_of_hidden_schedule():
    for change in [('position_gain',4.),('target_angular_velocity',[1.,0.,0.]),('new_parameter',True)]:
        p=sample();p['inputs'][change[0]]=change[1]
        with pytest.raises(ValueError):encode_packet(p)


def test_no_predicted_channel_disables_collision_callbacks_or_hard_limits():
    action=encode_packet(sample())
    for bit in [0.,1.]:
        action[:]=bit;action[SLICES['target_quaternion']]=[1.,0.,0.,0.]
        if bit:action[SLICES['orientation_axis_world']]=[0.,0.,1.]
        decoded=decode_action(action,0)
        assert all(decoded['safety_callbacks'].values())
        for key,value in FIXED_INPUTS.items():assert decoded['inputs'][key]==value
        assert decoded['inputs']['joint_limit_margin_rad']<=.08


def test_degenerate_nan_shape_and_disabled_axis_handling():
    action=encode_packet(sample())
    for invalid in [np.full(29,np.nan),np.zeros(28),np.zeros(29)]:
        with pytest.raises(ValueError):decode_action(invalid,0)
    action[SLICES['orientation_axis_world']]=0
    action[SLICES['orientation_axis_enabled']]=0
    assert decode_action(action,0)['inputs']['orientation_axis_world'] is None
    action[SLICES['orientation_axis_enabled']]=1
    with pytest.raises(ValueError):decode_action(action,0)


def test_model_absolute_action_config_does_not_mutate_legacy():
    old=dict(max_action_dim=32,max_state_dim=32,use_delta_action=True,
        output_features={'action':{'shape':[7]}},
        input_features={'observation.state':{'shape':[6]}})
    cfg=model_configuration(old)
    assert cfg['output_features']['action']['shape']==[ACTION_DIM]
    assert cfg['input_features']['observation.state']['shape']==[STATE_DIM]
    assert cfg['use_delta_action'] is False
    assert old['use_delta_action'] is True and old['output_features']['action']['shape']==[7]


def test_robot_state_only_contains_documented_proprioception_and_anchor():
    state=robot_state([1.]*7,[2.]*7,[3.]*3,[1.,0.,0.,0.],[4.]*7,[.04,.04])
    assert state.shape==(30,) and state.dtype==np.float32
    np.testing.assert_array_equal(state[:7],np.ones(7))
    np.testing.assert_array_equal(state[21:28],np.full(7,4.))


def test_stream_never_reuses_terminal_expert_mode_or_callback_schedule():
    p=sample();p['safety_callbacks']={k:None for k in p['safety_callbacks']}
    p['terminal_cartesian_hold']=True
    p2=copy.deepcopy(p);p2['frame']=1
    result=decode_stream(encode_stream([p,p2]))
    assert all(all(x['safety_callbacks'].values()) for x in result)
    assert all(x['terminal_cartesian_hold'] is False for x in result)


def test_runtime_reads_only_action_array_not_saved_upper_decisions(tmp_path,monkeypatch):
    import json
    import h5py
    from capstone_vla import hgwind_expert_trace as trace
    from capstone_vla.hf_archive import sha256
    from capstone_vla.ball_action_contract import SCHEMA
    capture=tmp_path/'capture.json';capture.write_text('not an input to decoder')
    source=tmp_path/'source.hdf5';source.write_bytes(b'provenance only')
    actions=tmp_path/'actions.npy'
    p=sample();p2=copy.deepcopy(p);p2['frame']=1
    np.save(actions,encode_stream([p,p2]))
    config=dict(schema='physical-vla-lab.robot-posture-configuration.v1',ik_offset_rad=[0.]*7,
        homotopy_enabled=False,homotopy_rank=0,strength=1.)
    plan=dict(cartesian_target_capture=str(capture),cartesian_target_capture_sha256=sha256(capture),
        source=str(source),source_reference_sha256=sha256(source),independent_packets=True,
        action_contract=SCHEMA,action_vectors=str(actions),action_vectors_sha256=sha256(actions),
        packet_gain_profile='fixed-9-8',packet_posture_profile='latched',robot_posture_configuration=config)
    def forbidden(*a,**kw):raise AssertionError('saved HDF5 actions were read')
    monkeypatch.setattr(h5py,'File',forbidden)
    try:
        trace.configure(plan)
        assert len(trace._STATE['packet_executor'].packets)==2
        actions.write_bytes(b'tampered')
        with pytest.raises(ValueError,match='Action labels changed'):trace.configure(plan)
    finally:
        trace._STATE.clear()


def test_motor_boundary_audit_rejects_hidden_settings_and_disabled_callback():
    from capstone_vla.ball_action_contract import SCHEMA
    p=sample();p2=copy.deepcopy(p);p2['frame']=1
    actions=encode_stream([p,p2])
    config=dict(schema='physical-vla-lab.robot-posture-configuration.v1',ik_offset_rad=[0.]*7,
        homotopy_enabled=False,homotopy_rank=0,strength=1.)
    initialization=dict(frame=0,configuration=config,initial_joint_position_rad=[0.]*7,
        lower_joint_limit_rad=[-3.]*7,upper_joint_limit_rad=[3.]*7,anchor_rad=[0.]*7)
    events=[]
    for packet in decode_stream(actions,[0.]*7):
        events.append(dict(frame=packet['frame'],inputs=dict(packet['inputs'],**packet['safety_callbacks']),
            gripper_closed=packet['gripper_closed'],latched_posture_context=dict(
                measured_joint_position_rad=[0.]*7,native_posture_target_rad=[0.]*7,
                native_posture_source='preflight_endpoint')))
    payload=dict(calls=events,posture_initialization=initialization)
    result=audit_contract(payload,actions)
    assert result['decoded_actions_match_motor_boundary'] and result['live_posture_verified']
    assert result['fixed_collision_callbacks_always_enabled']
    events[1]['inputs']['position_gain']=4.
    assert not audit_contract(payload,actions)['decoded_actions_match_motor_boundary']
    events[1]['inputs']['cartesian_velocity_projector']=None
    assert not audit_contract(payload,actions)['fixed_collision_callbacks_always_enabled']


def test_cross_environment_unit_vector_roundoff_only():
    from capstone_vla.ball_action_contract import native_inputs_match
    expected=sample()['inputs'];actual=copy.deepcopy(expected)
    actual['target_quaternion'][0]+=2.e-16
    assert actual!=expected and native_inputs_match(actual,expected)
    actual['orientation_axis_world']=[0.,0.,1.]
    expected['orientation_axis_world']=[0.,0.,1.+2.e-16]
    assert native_inputs_match(actual,expected)
    actual['target_position'][0]+=2.e-16
    assert not native_inputs_match(actual,expected)


def test_roundoff_comparison_rejects_meaningful_rotation_nan_null_and_callback_drift():
    from capstone_vla.ball_action_contract import native_inputs_match
    expected=sample()['inputs']
    for value in [[1.+1.e-12,0.,0.,0.],[float('nan'),0.,0.,0.],None,[1.,0.,0.]]:
        actual=copy.deepcopy(expected);actual['target_quaternion']=value
        assert not native_inputs_match(actual,expected)
    expected=dict(expected,cartesian_velocity_projector={'enabled':True})
    actual=copy.deepcopy(expected);actual['cartesian_velocity_projector']=None
    assert not native_inputs_match(actual,expected)


def test_packet_roundoff_does_not_allow_gripper_phase_or_constant_changes():
    from capstone_vla.ball_action_contract import decoded_packets_match
    packet=sample();actual=copy.deepcopy(packet)
    actual['inputs']['target_quaternion'][0]+=2.e-16
    assert decoded_packets_match([actual],[packet])
    for name,value in [('gripper_closed',True),('terminal_cartesian_hold',True),('frame',1)]:
        changed=copy.deepcopy(actual);changed[name]=value
        assert not decoded_packets_match([changed],[packet])
    actual['inputs']['position_gain']+=2.e-15
    assert not decoded_packets_match([actual],[packet])


def test_native_inplace_normalization_audit_copy_preserves_raw_records():
    from capstone_vla.ball_action_contract import normalization_audit_copy
    packet=sample();packet['inputs']['orientation_axis_world']=[0.,0.,1.]
    original=dict(calls=[dict(inputs=dict(packet['inputs'],**packet['safety_callbacks']))])
    original['calls'][0]['inputs']['orientation_axis_world']=[0.,0.,1.+2.e-16]
    before=copy.deepcopy(original)
    audited=normalization_audit_copy(original,[packet])
    assert original==before and audited!=original
    assert audited['calls'][0]['inputs']['orientation_axis_world']==[0.,0.,1.]


def test_native_normalization_audit_copy_cannot_hide_velocity_or_callback_change():
    from capstone_vla.ball_action_contract import normalization_audit_copy
    packet=sample();original=dict(calls=[dict(inputs=dict(packet['inputs'],**packet['safety_callbacks']))])
    original['calls'][0]['inputs']['target_linear_velocity']=[1.e-16,0.,0.]
    with pytest.raises(ValueError):normalization_audit_copy(original,[packet])
