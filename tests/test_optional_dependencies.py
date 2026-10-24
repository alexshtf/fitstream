import subprocess
import sys
import textwrap


def test_core_training_with_plotting_imports_blocked() -> None:
    script = textwrap.dedent("""
        import importlib.abc
        import sys

        class BlockPlottingImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'matplotlib', 'IPython'}:
                    raise ModuleNotFoundError(
                        f'Blocked optional dependency: {fullname}', name=fullname
                    )

        sys.meta_path.insert(0, BlockPlottingImports())

        import fitstream as fts
        import torch

        x = torch.tensor([[1.0], [2.0]])
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        events = fts.pipe(
            fts.epoch_stream((x, x), model, optimizer, torch.nn.MSELoss()),
            fts.take(1),
        )
        history = fts.collect(events)
        assert len(history) == 1
        assert history[0]['step'] == 1
        assert torch.isfinite(torch.tensor(history[0]['train_loss']))
        assert 'matplotlib' not in sys.modules
        assert 'IPython' not in sys.modules
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
