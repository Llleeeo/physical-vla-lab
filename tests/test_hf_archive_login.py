from pathlib import Path
import getpass
import shlex
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT=Path(__file__).resolve().parents[1]


def program():
    block=(ROOT/'scripts/hongtao_hf_archive.sh').read_text().split('  login)\n',1)[1].split('    ;;',1)[0]
    words=shlex.split(block)
    return words[words.index('-c')+1]


@pytest.mark.parametrize('account,accepted',[('LeeOoOo3',True),('other',False)])
def test_hidden_prompt_checks_owner_before_saving_and_never_writes_git(monkeypatch,capsys,account,accepted):
    save=Mock()
    monkeypatch.setattr(getpass,'getpass',lambda prompt:'test-only-not-a-token')
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(
        HfApi=lambda **kw:SimpleNamespace(whoami=lambda:{'name':account}),login=save))
    if accepted:
        exec(compile(program(),'<project-hf-login>','exec'),{})
        save.assert_called_once_with(token='test-only-not-a-token',add_to_git_credential=False)
    else:
        with pytest.raises(SystemExit,match='Wrong HF account'):
            exec(compile(program(),'<project-hf-login>','exec'),{})
        save.assert_not_called()
    assert 'test-only-not-a-token' not in capsys.readouterr().out
