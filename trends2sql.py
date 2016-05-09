import csv
import datetime
import re
import sys

class Reader:
    _daterange = re.compile("""(\d\d\d\d-\d\d-\d\d) - (\d\d\d\d-\d\d-\d\d)""")

    def __init__(self, stream):
        self._reader = csv.reader(stream)
        self._symbol = None

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            row = next(self._reader)
            if len(row) != 2:
                continue

            match = self._daterange.match(row[0])
            if match:
                start = datetime.datetime.strptime(match.group(1),
                    "%Y-%m-%d").date()
                end = datetime.datetime.strptime(match.group(2),
                    "%Y-%m-%d").date()
                value = int(row[1])
                return self._symbol, start, end, value
            elif row[0] == "Week":
                self._symbol = row[1].upper()

def process(stream):
    for symbol, start, end, value in Reader(stream):
        print("INSERT OR IGNORE INTO trends "
                "SELECT permno, '{}', '{}', {} "
                "FROM names "
                "WHERE tsymbol = '{}';".format(
            start.isoformat(),
            end.isoformat(),
            value,
            symbol))

def main(argv):
    print("BEGIN;")
    if len(argv) < 2:
        process(sys.stdin)
    else:
        for path in argv[1:]:
            with open (path) as stream:
                process(stream)
    print("END;")

if __name__ == "__main__":
    main(sys.argv)
