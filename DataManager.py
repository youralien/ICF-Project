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
        c.commit()

    def populateColumn(self, table, col_name, values):
        args = [(value, i+1) for i, value in enumerate(values)]        
        sql = 'UPDATE %s SET %s=? WHERE rowid=?'%(table, col_name)
        c = self.conn.cursor()
        c.executemany(sql, args)
        c.commit()

    def sqliteVersion(self):
        c = self.conn.cursor()
        c.execute('SELECT SQLITE_VERSION()')
        data = c.fetchone()
        print "SQLite version: %s" % data

    def __del__(self):
        self.conn.close()
        print "Closing connection to " + self.db

def main():
    db = 'Data/BKGDAT.db'
    table = 'Main'

    d = DataManager(db)
    d.sqliteVersion()
    # d.addColumn('Main', 'NormKeyday', 'float')
    # d.populateColumn(table, 'NormKeyday', [0,10,20,30])

if __name__ == '__main__':
    main()