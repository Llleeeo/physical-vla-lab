import copy
import pytest
from capstone_vla.ball_repeat import matched_config,outcome


def test_repeat_changes_only_seed_and_output_not_physics_or_controller():
    ref=dict(output='old',worker_output='old/worker',model_seed=90000,seed=345,
        approach_guard={'depth':5},checkpoint_hashes={'weights':'abc'},
        runtime_hashes={'/x/scripts/run_hgwind_closed_loop.py':'old','/x/controller.py':'same'})
    new=copy.deepcopy(ref);new.update(output='new',worker_output='new/worker',model_seed=190000)
    new['runtime_hashes']['/x/scripts/run_hgwind_closed_loop.py']='new'
    matched_config(ref,new,190000)
    for field,value in [('seed',346),('approach_guard',{'depth':0}),('checkpoint_hashes',{'weights':'other'})]:
        changed=copy.deepcopy(new);changed[field]=value
        with pytest.raises(ValueError):matched_config(ref,changed,190000)
    new['runtime_hashes']['/x/controller.py']='other'
    with pytest.raises(ValueError):matched_config(ref,new,190000)


def report():
    return dict(full_requested_integration_passed=True,initial_state_match=True,
        model_response_binding_verified=True,approach_audit={'verified':True},
        independent_grasp={'status':'fail'},runtime_safety_passed=True)


def test_integration_pass_does_not_count_as_grasp_and_aborts_remain_visible():
    r=report();assert outcome(r)=='grasp_fail'
    r['independent_grasp']['status']='pass';assert outcome(r)=='grasp_pass'
    r['initial_state_match']=False;assert outcome(r)=='incomplete_or_invalid'
    r['abort_reason']='velocity';assert outcome(r)=='runtime_abort'
    assert outcome({})=='incomplete_or_invalid'
