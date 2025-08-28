from trajanpy import Trajectory
import pytest


def test_trajectory_init():
    trajectory = Trajectory()
    assert not trajectory.has_frames
    assert trajectory.current_frame_index == 0


def test_trajectory_load_files(test_data_dir):
    trajectory = Trajectory()
    trajectory.load_files(
        [str(test_data_dir / "coord.pdb"), str(test_data_dir / "traj.dcd")]
    )
    assert trajectory.has_frames


def test_trajectory_next_frame(test_data_dir):
    trajectory = Trajectory()
    trajectory.load_files(
        [str(test_data_dir / "coord.pdb"), str(test_data_dir / "traj.dcd")]
    )
    assert trajectory.next_frame()
    assert trajectory.current_frame_index == 1
    assert trajectory.next_frame()
    assert trajectory.current_frame_index == 2


def test_trajectory_update_topology(test_data_dir):
    trajectory =Trajectory()
    trajectory.load_files(
        [str(test_data_dir / "coord.pdb"), str(test_data_dir / "traj.dcd")]
    )
    trajectory.next_frame()
    trajectory.update_topology()
    assert trajectory.topology.num_bonds() > 0


def test_trajectory_molecules(test_data_dir):
    trajectory = Trajectory()
    #trajectory.load_files(
    #    [str(test_data_dir / "coord.pdb"), str(test_data_dir / "traj.dcd")]
    #)
    trajectory.load_files(
        [str(test_data_dir / "coord.pdb"),]
    )
    trajectory.next_frame()
    molecules = trajectory.extract_molecules()
    assert len(molecules) > 0
