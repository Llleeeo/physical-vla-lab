from pathlib import Path
import importlib.util
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('depth_worker', ROOT/'scripts/run_hgwind_depth_worker.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def probe(depth=.002, projection=None):
    class Robot:
        def reset(self): self._up = np.array([0.,0.,1.])
        def _approach_action(self, goal, quat, dt, p_ee, **kwargs): return goal, kwargs
        def _environment_safe_translation(self, p, q, v, **kw):
            return v if projection is None else np.asarray(projection)
    module.install_depth_bias(Robot, depth)
    robot=Robot();robot.reset();return robot


def test_only_goal_moves_down_and_input_and_semantics_preserved():
    robot=probe();goal=np.array([1.,2.,3.]);result,kw=robot._approach_action(goal,[1,0,0,0],1/60,[0,0,0],command_velocity=[.1,0,0])
    np.testing.assert_allclose(result,[1,2,2.998]);np.testing.assert_array_equal(goal,[1,2,3]);assert kw['command_velocity']==[.1,0,0]


def test_collision_filter_can_block_depth():
    robot=probe(projection=[0,0,0]);result,_=robot._approach_action([1,2,3],[1,0,0,0],1/60,[0,0,0]);np.testing.assert_array_equal(result,[1,2,3]);assert robot._capstone_depth_stats['limited_calls']==1


def test_projection_never_adds_lateral_motion_or_more_than_requested():
    robot=probe(projection=[10,10,-10]);result,_=robot._approach_action([1,2,3],[1,0,0,0],1/60,[0,0,0]);np.testing.assert_allclose(result,[1,2,2.998])


def test_zero_depth_is_exact_passthrough():
    robot=probe(depth=0);goal=[1,2,3];result,_=robot._approach_action(goal,[1,0,0,0],1/60,[0,0,0]);assert result is goal


def test_deeper_than_two_mm_rejected():
    with pytest.raises(ValueError):probe(depth=.01)
