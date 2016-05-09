"""
Stocks database.
"""

import collections
import itertools
import logging
import sqlite3
import util

class Stocks:
    """Stocks data generator."""

    # feature vector definition
    Features = collections.namedtuple("Features",
        ["askhi", "bidlo", "ret", "vol", "ask", "bid", "retx", "trend"])

    # default features are all zero
    default_features = Features._make([0] * len(Features._fields))

    def __init__(self, path):
        """Open database.

        Parameters:
        path -- database path
        """

        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row

    def close(self):
        """Close database."""
        self._db.close()

    def permno(self, symbol):
        """Return the CRSP permno for a symbol."""

        # query data
        cursor = self._db.cursor()
        cursor.execute("""
            SELECT permno
            FROM names
            WHERE tsymbol = :symbol
            ORDER BY date DESC
            LIMIT 1
            """, dict (symbol = symbol))

        row = cursor.fetchone()
        cursor.close()
        return row["permno"] if row else None

    def permnos(self, symbols):
        """Return the CRSP permnos for a sequence of symbols."""

        return [self.permno(symbol) for symbol in symbols]

    def timeseries(self, permno, start = None, end = None):
        """Return the timeseries of features for an issue.

        Parameters:
        permno -- CRSP permno
        start -- start date (None = start of time)
        end -- end date (None = end of time)
        """

        # query data
        cursor = self._db.cursor()
        cursor.execute("""
            SELECT prices.date,
                prices.askhi,
                prices.bidlo,
                prices.ret,
                prices.vol,
                prices.ask,
                prices.bid,
                prices.retx,
                trends.trend
            FROM prices LEFT JOIN trends
            ON prices.permno = trends.permno
                AND prices.date >= trends.start_date
                AND prices.date <= trends.end_date
            WHERE prices.permno = :permno
                AND prices.date >= :start
                AND prices.date <= :end
            ORDER BY prices.date
            """, dict(
                permno = permno,
                start = start or "0000-00-00",
                end = end or "9999-99-99"))

        # constuct features from data rows
        row = cursor.fetchone()
        lastrow = row
        while row:
            yield row["date"], self.Features(
                askhi = self._change(row["askhi"], lastrow["askhi"]),
                bidlo = self._change(row["bidlo"], lastrow["bidlo"]),
                vol = self._change(row["vol"], lastrow["vol"]),
                ret = util.tofloat(row["ret"]),
                ask = self._change(row["ask"], lastrow["ask"]),
                bid = self._change(row["bid"], lastrow["bid"]),
                retx = util.tofloat(row["retx"]),
                trend = util.tofloat(row["trend"]) / 100)
            lastrow = row
            row = cursor.fetchone()

        cursor.close()

    def multiseries(self, permnos, start = None, end = None):
        """Return the timeseries of features for a set of issues.

        Parameters:
        permnos -- sequence of CRSP permnos
        start -- start date (None = start of time)
        end -- end date (None = end of time)
        """

        return self._synchronize(
            [self.timeseries(p, start = start, end = end) for p in permnos])

    def _change(self, x, y):
        """Return the relative change between x and y."""

        if None in (x, y):
            return 0
        change = (x - y) / y if y else 0
        return min(max(change, -1), 1)

    def _synchronize(self, timeseries_sequence):
        """Synchronize a sequence of timeseries."""

        # join sequences on date
        for items in util.join(timeseries_sequence, key = lambda x: x[0]):
            # construct a row with one date and default vectors as needed
            date = next(filter(None, items))[0]
            values = list(itertools.chain.from_iterable(
                [i[1:] if i else [self.default_features] for i in items]))
            yield [date] + values
