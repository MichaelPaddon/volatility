CRSP = crsp20150930.csv.gz
TRENDS = $(wildcard trends/*.csv)
RESULTS =\
	 results/alpha.csv\
	 results/beta.csv\
	 results/gamma.csv\
	 results/delta.csv\
	 results/epsilon.csv\
	 results/zeta.csv\
	 results/eta.csv\
	 results/theta.csv\
	 results/iota.csv\
	 results/kappa.csv\
	 results/mu.csv\
	 results/xi.csv\
	 results/omnicron.csv\
	 results/pi.csv

all: $(RESULTS)

results/%.csv: stocks.db
	mkdir -p results
	time python model.py $(basename $(notdir $@)) > $@.tmp
	mv $@.tmp $@

stocks.db: $(CRSP) $(TRENDS)
	sqlite3 $@ < schema.sql
	zcat $(CRSP) | python3 crsp2sql.py | sqlite3 $@
	python3 trends2sql.py $(TRENDS) | sqlite3 $@
