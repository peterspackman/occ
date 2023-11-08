#include <occ/core/log.h>
#include <occ/core/sqlite_datastore.h>

namespace occ::core::sqlite {

Connection::Connection(const std::string &database, int flags)
    : db(0), flags(flags) {
    err = sqlite3_open_v2(database.c_str(), &db, flags, NULL);
    occ::log::trace("Err flag set in Connection constructor: {}", err);
}

Connection::~Connection() { close(true); }

bool Connection::is_open() const { return db != 0; }

bool Connection::is_ok() const { return error_code() == SQLITE_OK; }

bool Connection::close(bool force) {
    // TODO: force

    if (!db)
        return true;

    if ((err = sqlite3_close(db)) != SQLITE_OK)
        return false;

    return true;
}

Cursor *Connection::cursor() {
    if (is_open() && is_ok())
        return new sqlite::Cursor(this);
    return 0;
}

int Connection::error_code() const { return err; }

const char *Connection::error_message() const { return sqlite3_errstr(err); }

sqlite3_int64 Connection::last_insert_rowid() const {
    return sqlite3_last_insert_rowid(db);
}

Row::Row() : cursor(0) {}

Row::Row(sqlite::Cursor *c) : cursor(c) {}

Row::Row(const Row &other) : cursor(other.cursor) {}

Row &Row::operator=(Row other) {
    swap(*this, other);
    return *this;
}

void Row::swap(Row &lhs, Row &rhs) { std::swap(lhs.cursor, rhs.cursor); }

Row::operator bool() const { return cursor != 0; }

int Row::column_count() const {
    if (!cursor)
        return 0;

    return sqlite3_column_count(cursor->stmt);
}

int Row::column_type(int col) const {
    if (!cursor)
        return 0;

    return sqlite3_column_type(cursor->stmt, col);
}

const char *Row::column_name(int col) const {
    if (!cursor)
        return 0;

    return sqlite3_column_name(cursor->stmt, col);
}

int Row::get_bytes(int col) const {
    if (!cursor)
        return 0;

    return sqlite3_column_bytes(cursor->stmt, col);
}

int Row::get_int(int col) const {
    if (!cursor)
        return 0;

    return sqlite3_column_int(cursor->stmt, col);
}

const char *Row::get_text(int col) const {
    if (!cursor)
        return 0;

    return (const char *)sqlite3_column_text(cursor->stmt, col);
}

Cursor::Cursor(Connection *conn)
    : conn(conn), stmt(0), err(SQLITE_OK), state(-1) {}

Cursor::~Cursor() { finalize(); }

bool Cursor::execute(const std::string &statement) {
    if (!prepare(statement))
        return false;

    Row row = next();

    return err == SQLITE_DONE || err == SQLITE_ROW;
}

bool Cursor::execute(const std::string &statement, const char *bindfmt, ...) {
    va_list ap;

    if (!prepare(statement))
        return false;

    occ::log::trace("start binding sqlite3::Cursor");

    va_start(ap, bindfmt);
    bool ret = bindv(bindfmt, ap);
    va_end(ap);

    if (!ret) {
        occ::log::trace("failed binding sqlite3::Cursor");
        return false;
    }

    Row row = next();

    return err == SQLITE_DONE || err == SQLITE_ROW;
}

Row Cursor::next() {

    while (true) {
        if (state == -1) {
            state = 0;
            err = sqlite3_step(stmt);
        } else if (state == 1) {
            err = sqlite3_step(stmt);
        } else if (state == 0) {
            state = 1;
        }

        if (col_names.empty() && state == 0 && err == SQLITE_ROW) {
            fetch_column_names(Row(this));
        }

        if (err == SQLITE_DONE) {
            return Row();
        } else if (err == SQLITE_ROW) {
            return Row(this);
        } else if (err == SQLITE_BUSY) {
            continue;
        } else {
            occ::log::debug("Unknown state: err={}", err);
            return Row();
        }
    }
}

Connection *Cursor::get_connection() const { return conn; }

bool Cursor::prepare(const std::string &statement) {
    occ::log::trace("sqlite stmt: '{}'", statement);

    finalize();

    col_names.clear();

    state = -1;

    sql = statement;

    int rc = sqlite3_prepare_v2(conn->db, sql.c_str(), sql.size(), &stmt, NULL);

    occ::log::trace("err: {} {}", rc, sqlite3_errstr(rc));

    return rc == SQLITE_OK;
}

bool Cursor::bindv(const char *fmt, va_list ap) {
    char c;
    int altformat = 0;
    int idx = 0;
    int len = -1;

    enum { ALT_L = 0x1, ALT_WIDTH = 0x2 };

    occ::log::trace("sqlite bindfmt: '%s'", fmt);

    while ((c = *fmt++) != '\0') {
        len = -1;
        altformat = 0;

        // skip whitespace
        if (isspace(c))
            continue;

        // if there's something other than %
        if (c != '%')
            goto literal;

    again:
        switch (c = *fmt++) {
        // %%
        case '%':
        literal:
            return false;

        // field width
        case '.':
            altformat |= ALT_WIDTH;
            c = *fmt++;
            if (c == '*') {
                len = va_arg(ap, int);
                // DEBUG_FMT("field width from next arg: %d", len);
                goto again;
            } else if (isdigit(c)) {
                char *endptr;
                len = strtol(fmt - 1, &endptr, 10);
                fmt = endptr;
                // DEBUG_FMT("field width from fmtstr: %d", len);
                goto again;
            }
            return false;

        // int64 specifier for 'd'
        // utf16 specifier for 's'
        case 'l':
            // DEBUG_STR("ALT_L");
            altformat |= ALT_L;
            goto again;

        // int / int64
        case 'd':
            if (altformat & (~ALT_L)) {
                int val = va_arg(ap, int);
                sqlite3_bind_int(stmt, idx++, val);
            } else {
                sqlite3_int64 val = va_arg(ap, sqlite3_int64);
                sqlite3_bind_int64(stmt, idx++, val);
            }
            continue;

        // double
        case 'f': {
            double val = va_arg(ap, double);
            sqlite3_bind_double(stmt, idx++, val);
        }
            continue;

        // utf-8 string
        case 's': {
            if (altformat & ALT_L) {
                void *val = va_arg(ap, void *);
                // DEBUG_STR("binding utf-16 str");
                sqlite3_bind_text16(stmt, idx++, val, len, SQLITE_TRANSIENT);
            } else {
                char *val = va_arg(ap, char *);
                // DEBUG_FMT("binding utf-8 str <%.*s>", (len < 0) ? strlen(val)
                // : len, val);
                sqlite3_bind_text(stmt, idx++, val, len, SQLITE_TRANSIENT);
            }
        }
            continue;

        // zeroblob
        case 'z':
            // DEBUG_FMT("binding zeroblob of len %d", len);
            sqlite3_bind_zeroblob(stmt, idx++, len);
            continue;

        // null
        case 'n':
            // DEBUG_STR("binding null");
            sqlite3_bind_null(stmt, idx++);
            continue;

        // blob
        // case 'b':
        //     DEBUG_STR("binding blob");
        //     sqlite3_bind_blob(stmt, idx++, )

        // unknown conversion specifier
        default:
            return false;
        }
    }

    return true;
}

bool Cursor::bind(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    bool ret = bindv(fmt, ap);
    va_end(ap);
    return ret;
}

void Cursor::finalize() {
    if (stmt) {
        occ::log::trace("finalizing previous stmt");
        sqlite3_finalize(stmt);
        stmt = 0;
    }
}

void Cursor::fetch_column_names(Row row) {
    col_names.clear();
    for (int i = 0; i < row.column_count(); ++i) {
        const char *p = row.column_name(i);
        std::string name;
        if (p)
            name.assign(p);
        col_names.push_back(name);
        occ::log::trace("sqlite colname: {}: {}", i, p);
    }
}

bool Cursor::reset() { return sqlite3_reset(stmt) == SQLITE_OK; }

bool Cursor::clear_bindings() {
    return sqlite3_clear_bindings(stmt) == SQLITE_OK;
}

} // namespace occ::core::sqlite
