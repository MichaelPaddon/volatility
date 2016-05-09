import collections
import csv
import datetime
import sys
import util

class Reader:
    def __init__(self, stream):
        self._reader = csv.reader(stream)
        header = next(self._reader, [])
        self._Row = collections.namedtuple("Row", [f.lower() for f in header])

    def __iter__(self):
        return self

    def __next__(self):
        return self._Row(*next(self._reader))

    @property
    def Row(self):
        return self._Row

def process(stream):
    name_fields = ("ncusip", "naics", "ticker", "siccd",
        "comnam", "exchcd", "primexch", "secstat",
        "shrcls", "shrcd", "trdstat", "tsymbol")

    price_fields = ("askhi", "bidlo", "ret", "prc", "vol",
        "retx", "openprc", "ask", "bid", "numtrd")

    permnos = set()
    lastnames = None
    reader = Reader(stream)
    for row in reader:
        date = datetime.datetime.strptime(row.date, "%Y/%m/%d").date()

        if row.permno not in permnos:
            print("INSERT OR REPLACE INTO issues VALUES({},{},{});".format(
                row.permno or "NULL",
                row.permco or "NULL",
                row.issuno or "NULL"))
            permnos.add(row.permno)

        names = [getattr(row, f) for f in name_fields]
        if any(names) and names != lastnames:
            print("INSERT OR REPLACE INTO names "
                    "VALUES({},{},{},{},{},{},{},{},{},{},{},{},{},{});".format(
                row.permno or "NULL",
                util.singlequote(date.isoformat()),
                util.singlequote(row.ncusip) if row.ncusip else "NULL",
                row.naics or "NULL",
                util.singlequote(row.ticker) if row.ticker else "NULL",
                util.singlequote(row.siccd) if row.siccd else "NULL",
                util.singlequote(row.comnam) if row.comnam else "NULL",
                row.exchcd or "NULL",
                util.singlequote(row.primexch) if row.primexch else "NULL",
                util.singlequote(row.secstat) if row.secstat else "NULL",
                util.singlequote(row.shrcls) if row.shrcls else "NULL",
                row.shrcd or "NULL",
                util.singlequote(row.trdstat) if row.trdstat else "NULL",
                util.singlequote(row.tsymbol) if row.tsymbol else "NULL"))
            lastnames = names

        prices = [getattr(row, f) for f in price_fields]
        if any(prices):
            print("INSERT OR REPLACE INTO prices "
                    "VALUES({},{},{},{},{},{},{},{},{},{},{},{});".format(
                row.permno or "NULL",
                util.singlequote(date.isoformat()),
                row.askhi or "NULL",
                row.bidlo or "NULL",
                row.ret if util.isfloat(row.ret)\
                    else util.singlequote(row.ret) if row.ret else "NULL",
                row.prc or "NULL",
                row.vol or "NULL",
                row.retx if util.isfloat(row.retx)\
                    else util.singlequote(row.retx) if row.retx else "NULL",
                row.openprc or "NULL",
                row.ask or "NULL",
                row.bid or "NULL",
                row.numtrd or "NULL"))

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
