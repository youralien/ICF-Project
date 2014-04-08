import sqlite3 as lite

class DataManager():
    def __init__(self, db):
        self.db = db
        print "Opening connection to " + self.db
        self.conn = lite.connect(self.db)

    def addColumn(self, table, col_name, col_type):
        args = (table, col_name, col_type)
        sql = 'ALTER TABLE %s ADD COLUMN %s %s'%(table, col_name, col_type)
        c = self.conn.cursor()
        c.execute(sql)
        self.conn.commit()

    def populateColumn(self, table, col_name, values):
        args = [(value, i+1) for i, value in enumerate(values)]        
        sql = 'UPDATE %s SET %s=? WHERE rowid=?'%(table, col_name)
        c = self.conn.cursor()
        c.executemany(sql, args)
        self.conn.commit()

    def sqliteVersion(self):
        c = self.conn.cursor()
        c.execute('SELECT SQLITE_VERSION()')
        data = c.fetchone()
        print "SQLite version: %s" % data

    def __del__(self):
        self.conn.close()
        print "Closing connection to " + self.db

def main():
    # sqlite3 BKGDAT.db
    # create table Main(DATE text, FLT integer, ORG text, DES text, CAP integer, BC text, BKD integer, AVAIL integer, AUTH integer, KEYDAY integer, TOTALBKD integer)
    # .separator ","
    # import BKGDAT_MOAR_Filtered.txt Main
    db = 'Data/BKGDAT.db'
    table = 'Main'

    d = DataManager(db)
    d.sqliteVersion()

    import pandas as pd
    df = pd.read_csv('Data/Normalized_BKGDAT_Filtered.txt')

    print df['KEYDAY']

    #d.addColumn('Main', 'NORM_KEYDAY', 'float')
    d.populateColumn('Main', 'NORM_KEYDAY', list(df['KEYDAY']))

if __name__ == '__main__':
    main()