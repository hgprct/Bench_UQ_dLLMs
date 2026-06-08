"""Tests for Pareto frontier computation in report."""

import pytest

from cli.report import _compute_pareto_front, _load_latency_csv


class TestParetoFront:
    def test_simple_frontier(self):
        points = [(0.0, 0.8), (1.0, 0.9), (2.0, 0.85), (0.5, 0.7)]
        front = _compute_pareto_front(points)
        assert set(front) == {0, 1}

    def test_all_on_frontier(self):
        points = [(0.0, 0.5), (1.0, 0.7), (2.0, 0.9)]
        front = _compute_pareto_front(points)
        assert set(front) == {0, 1, 2}

    def test_single_point(self):
        points = [(1.0, 0.8)]
        front = _compute_pareto_front(points)
        assert front == [0]

    def test_empty(self):
        assert _compute_pareto_front([]) == []

    def test_dominated_points_excluded(self):
        points = [(0.0, 0.9), (0.0, 0.5), (1.0, 0.95)]
        front = _compute_pareto_front(points)
        assert 0 in front
        assert 2 in front
        assert 1 not in front


class TestLoadLatency:
    def test_loads_csv_correctly(self, tmp_path):
        csv_content = (
            "run_id,model_id,dataset,family,feature,n_prompts,n_extra_gen,n_nli_pairs,"
            "time_gen_mean,time_gen_std,time_nli_mean,time_nli_std,"
            "time_math_mean,time_math_std,time_total_mean,time_total_std,overhead_mean,overhead_std\n"
            "test,m,d,token,msp,50,0,0,1.5,0.1,0.0,0.0,0.001,0.0,1.501,0.1,0.0,0.1\n"
            "test,m,d,iid,ecc,50,19,400,8.0,0.5,2.0,0.3,0.01,0.0,10.01,0.7,8.509,0.7\n"
        )
        csv_path = tmp_path / "latency_summary.csv"
        csv_path.write_text(csv_content)

        latency = _load_latency_csv(csv_path)
        assert "msp" in latency
        assert "ecc" in latency
        assert latency["msp"]["overhead_mean"] == pytest.approx(0.0)
        assert latency["ecc"]["overhead_mean"] == pytest.approx(8.509)
        assert latency["ecc"]["overhead_std"] == pytest.approx(0.7)
        assert latency["ecc"]["family"] == "iid"

    def test_loads_csv_without_overhead_std(self, tmp_path):
        csv_content = (
            "run_id,model_id,dataset,family,feature,n_prompts,n_extra_gen,n_nli_pairs,"
            "time_gen_mean,time_gen_std,time_nli_mean,time_nli_std,"
            "time_math_mean,time_math_std,time_total_mean,time_total_std,overhead_mean\n"
            "test,m,d,token,msp,50,0,0,1.5,0.1,0.0,0.0,0.001,0.0,1.501,0.1,0.0\n"
        )
        csv_path = tmp_path / "latency_summary.csv"
        csv_path.write_text(csv_content)

        latency = _load_latency_csv(csv_path)
        assert latency["msp"]["overhead_std"] == pytest.approx(0.0)
