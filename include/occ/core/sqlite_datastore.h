#pragma once
#include <sqlite3.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace occ::core::sqlite {

struct Cursor;

struct Connection {
    explicit Connection(const std::string &filename,
                        int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE);
    ~Connection();

    bool is_open() const;
    bool is_ok() const;
    bool close(bool force = false);

    Cursor *cursor();

    int error_code() const;
    const char *error_message() const;

    sqlite3_int64 last_insert_rowid() const;

    sqlite3 *db;
    int flags{0};
    int err{0};

  private:
    Connection();
    Connection(const Connection &);
};

struct Row {
    Row();
    Row(Cursor *c);
    Row(const Row &other);

    Row &operator=(Row other);

    void swap(Row &lhs, Row &rhs);

    operator bool() const;

    int column_count() const;

    int column_type(int col) const;

    const char *column_name(int col) const;

    int get_bytes(int col) const;
    int get_int(int col) const;
    const char *get_text(int col) const;

    // int get_int(const std::string &name);
    // const char *get_text(const std::string &name);

    Cursor *cursor;
};

struct Cursor {
    friend class Connection;
    ~Cursor();

    bool execute(const std::string &statement);
    bool execute(const std::string &statement, const char *bindfmt, ...);

    Row next();

    bool prepare(const std::string &statement);
    bool bindv(const char *fmt, va_list);
    bool bind(const char *fmt, ...);

    bool reset();

    bool clear_bindings();

    Connection *get_connection() const;

    Connection *conn;
    sqlite3_stmt *stmt;
    int err;
    std::string sql;
    int state;

  private:
    Cursor(Connection *conn);

    void finalize();
    void fetch_column_names(Row row);

    std::vector<std::string> col_names;

    friend struct connection;
};

} // namespace occ::core::sqlite
