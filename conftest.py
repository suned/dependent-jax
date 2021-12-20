from typing import Dict
import tokenize
import re
import difflib

import pytest
import _pytest
from mypy.api import run


def pytest_collect_file(parent, path):
    if path.ext == ".pyi" and path.basename.startswith("test"):
        return StubFile.from_parent(parent, fspath=path)


class StubFile(pytest.File):
    def check_file(self, filename: str) -> Dict[int, str]:
        pattern = r'(?P<file>.+):(?P<line>\d+): (?P<msg>.+)'
        end_pattern = r'Found \d+ errors? in \d+ file \(checked \d+ source files?\)'
        stdout, stderr, _ = run([filename, '--show-traceback', '--raise-exceptions'])
        lines = {}
        for line in stdout.splitlines() + stderr.splitlines():
            if re.match(end_pattern, line):
                continue
            m = re.match(pattern, line)
            lines[int(m.group('line'))] = m.group('msg')
        return lines

    def get_comments(self, filename: str):
        comments = {}
        with open(filename) as f:
            for toktype, tok, start, end, line in tokenize.generate_tokens(f.readline):
                if toktype == tokenize.COMMENT and re.search(r'^# ?note|error: .+', tok):
                    comments[start[0]] = re.sub(r'^# ?', '', tok)
        return comments

    def collect(self):
        output = self.check_file(str(self.fspath))
        comments = self.get_comments(str(self.fspath))
        for line, msg in output.items():
            name = f'{self.fspath.basename}:{line}'
            if line not in comments:
                yield StubItem.from_parent(parent=self, name=name, line=line, msg=msg, comment='')
            else:
                yield StubItem.from_parent(parent=self, name=name, line=line, msg=msg, comment=comments[line])
        for line, comment in comments.items():
            if line in output:
                continue
            name = f'{self.fspath.basename}:{line}'
            yield StubItem.from_parent(parent=self, name=name, line=line, msg='', comment=comment)


class MypyRepr(_pytest._code.code.TerminalRepr):
    def __init__(self, msg):
        self.msg = msg

    def toterminal(self, tw):
        tw.line(self.msg, bold=True, red=True)


class StubItem(pytest.Item):
    def __init__(self, name, parent, msg, comment, line):
        super().__init__(name, parent)
        self.msg = msg
        self.comment = comment
        self.line = line

    def runtest(self):
        if self.msg == '':
            error_msg = f'No output on line {self.line}'
        else:
            error_msg = f"Unexpected mypy output on line {self.line}"
        assert self.msg == self.comment, error_msg

    def repr_failure(self, excinfo, *args, **kwargs):
        return MypyRepr(excinfo.exconly())

    def reportinfo(self):
        return self.fspath, 0, self.name