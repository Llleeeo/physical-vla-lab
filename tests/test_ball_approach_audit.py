import copy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('ball_analysis',Path(__file__).resolve().parents[1]/'scripts/analyze_hgwind_closed_loop.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_roundoff_accepted_but_geometry_and_discrete_changes_rejected():
    expected=dict(frame=0,valid=True,mode='aligned_pass',changed_channels=[2],position=[.14,-4.3])
    recorded=copy.deepcopy(expected);recorded['position'][0]+=1e-15
    module.audit_approach_value(expected,recorded)
    for key,value in [('frame',1),('valid',False),('mode','missing_pose_hold'),('changed_channels',[9]),('position',[.140001,-4.3])]:
        changed=copy.deepcopy(recorded);changed[key]=value
        with pytest.raises(ValueError):module.audit_approach_value(expected,changed)
    with pytest.raises(ValueError):module.audit_approach_value(expected,dict(recorded,extra=1))
    with pytest.raises(ValueError):module.audit_approach_value(1,True)
    with pytest.raises(ValueError):module.audit_approach_value(float('nan'),float('nan'))


def test_only_named_legacy_analyzer_can_be_revised(tmp_path):
    result={};module.verify_analysis_revision(module.__file__,module.LEGACY_APPROACH_ANALYZER_SHA256,result)
    assert 'posthoc_analyzer_correction' in result
    other=tmp_path/'controller.py';other.write_text('unchanged')
    with pytest.raises(ValueError):module.verify_analysis_revision(other,module.LEGACY_APPROACH_ANALYZER_SHA256,{})
    with pytest.raises(ValueError):module.verify_analysis_revision(module.__file__,'not-an-approved-original',{})


def test_nonlinear_fit_roundoff_only_with_explicit_near_capture_tolerance():
    old={'valid':True,'velocity':[.01]};new=copy.deepcopy(old);new['velocity'][0]+=6e-12
    with pytest.raises(ValueError):module.audit_approach_value(old,new)
    module.audit_approach_value(old,new,atol=module.NEAR_CAPTURE_AUDIT_ATOL)
    new['velocity'][0]+=.000001
    with pytest.raises(ValueError):module.audit_approach_value(old,new,atol=module.NEAR_CAPTURE_AUDIT_ATOL)
    with pytest.raises(ValueError):module.audit_approach_value(True,False,atol=module.NEAR_CAPTURE_AUDIT_ATOL)
    module.verify_analysis_revision(module.__file__,module.NEAR_CAPTURE_ANALYZER_SHA256,{})
