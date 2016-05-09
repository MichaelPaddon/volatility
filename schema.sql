-- CRSP Issues
CREATE TABLE IF NOT EXISTS issues (
	permno INTEGER PRIMARY KEY NOT NULL,	-- permanent issue number
	permco INTEGER NOT NULL,		-- permanent company number
	issuno INTEGER				-- NASDAQ issue number
);
CREATE INDEX IF NOT EXISTS issues_permco_idx ON issues(permco);

-- CRSP Names
CREATE TABLE IF NOT EXISTS names (
	permno INTEGER NOT NULL			-- permanent issue number
		REFERENCES issues(permno),
	date TEXT NOT NULL,			-- date effective
        ncusip INTEGER,				-- CUISP code
        naics INTEGER,				-- NAICS code
	ticker TEXT,				-- ticker symbol
	siccd TEXT,				-- SIC code
	comnam TEXT NOT NULL,			-- company name
	exchcd INTEGER,				-- company name
	primexch TEXT,				-- primary exchange
	secstat TEXT NOT NULL,			-- security status
 	shrcls TEXT,				-- share class
 	shrcd INTEGER NOT NULL,			-- share code
	trdstat TEXT NOT NULL,			-- trading status
	tsymbol TEXT,				-- trading symbol
        UNIQUE(permno, date)
);
CREATE INDEX IF NOT EXISTS names_permno_idx ON names(permno);
CREATE INDEX IF NOT EXISTS names_date_idx ON names(date);
CREATE INDEX IF NOT EXISTS names_ticker_idx ON names(ticker);
CREATE INDEX IF NOT EXISTS names_tsymbol_idx ON names(tsymbol);

-- CRSP Prices
CREATE TABLE IF NOT EXISTS prices (
	permno INTEGER NOT NULL			-- permanent issue number
		REFERENCES issues(permno),
	date TEXT NOT NULL,			-- date
	askhi REAL,				-- ask or high price
	bidlo REAL,				-- bid or low price
        ret REAL,				-- holding period total return
	prc REAL,				-- price or bid/ask average
	vol INTEGER,				-- volume traded
        retx REAL,				-- return without dividends
	openprc REAL,				-- open price
	ask REAL,				-- closing ask
	bid REAL,				-- closing bid
        numtrd INTEGER,				-- nasdaq number of trades
        UNIQUE(permno, date)
);
CREATE INDEX IF NOT EXISTS prices_permno_idx ON prices(permno);
CREATE INDEX IF NOT EXISTS prices_date_idx ON prices(date);

-- Google Trends
CREATE TABLE IF NOT EXISTS trends (
	permno INTEGER NOT NULL			-- permanent issue number
		REFERENCES issues(permno),
	start_date TEXT NOT NULL,		-- start date
	end_date TEXT NOT NULL,			-- end date
        trend INTEGER,				-- relative percentage activity
	UNIQUE(permno, start_date, end_date)
);
CREATE INDEX IF NOT EXISTS trends_permno_idx ON trends(permno);
CREATE INDEX IF NOT EXISTS trends_start_date_idx ON trends(start_date);
CREATE INDEX IF NOT EXISTS trends_end_date_idx ON trends(end_date);
