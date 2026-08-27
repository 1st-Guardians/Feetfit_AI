import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.io.PrintWriter;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Locale;
import java.util.logging.Logger;

import javax.sql.DataSource;

import org.flywaydb.core.Flyway;
import org.flywaydb.core.api.MigrationInfo;
import org.flywaydb.core.api.MigrationInfoService;
import org.flywaydb.core.api.MigrationVersion;
import org.flywaydb.core.api.output.MigrateResult;
import org.flywaydb.core.api.output.ValidateResult;

/** Exact-target production Flyway runner. Python owns all authority checks. */
public final class FeetfitFlywayProductionApply {
    private static final String AUTHORIZED_DATABASE = "feetfit";
    private static final String AUTHORIZED_QUERY =
            "serverTimezone=Asia%2FSeoul&characterEncoding=UTF-8&useSSL=false&allowPublicKeyRetrieval=true";

    private static String requiredEnv(String name) {
        String value = System.getenv(name);
        if (value == null || value.isBlank()) {
            throw new IllegalStateException("missing_environment_" + name);
        }
        return value;
    }

    private static String databaseFromJdbc(String jdbcUrl) {
        String prefix = "jdbc:mysql://";
        if (!jdbcUrl.startsWith(prefix)) {
            throw new IllegalStateException("non_mysql_jdbc_url");
        }
        int slash = jdbcUrl.indexOf('/', prefix.length());
        if (slash < 0 || slash == jdbcUrl.length() - 1) {
            throw new IllegalStateException("jdbc_database_missing");
        }
        int query = jdbcUrl.indexOf('?', slash + 1);
        return query < 0 ? jdbcUrl.substring(slash + 1) : jdbcUrl.substring(slash + 1, query);
    }

    private static void assertCanonicalJdbcUrl(String jdbcUrl) {
        int query = jdbcUrl.indexOf('?');
        if (query < 0 || !AUTHORIZED_QUERY.equals(jdbcUrl.substring(query + 1))) {
            throw new IllegalStateException("non_canonical_jdbc_query");
        }
    }

    private static String jsonEscape(String value) {
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static void writeResult(
            Path resultPath,
            String status,
            String stage,
            int migrationsExecuted,
            int pendingBefore,
            int pendingAfter,
            String currentVersion,
            String errorClass) throws Exception {
        String json = "{" +
                "\"status\":\"" + jsonEscape(status) + "\"," +
                "\"stage\":\"" + jsonEscape(stage) + "\"," +
                "\"migrationsExecuted\":" + migrationsExecuted + "," +
                "\"pendingBefore\":" + pendingBefore + "," +
                "\"pendingAfter\":" + pendingAfter + "," +
                "\"currentVersion\":\"" + jsonEscape(currentVersion) + "\"," +
                "\"errorClass\":\"" + jsonEscape(errorClass) + "\"" +
                "}";
        Files.createDirectories(resultPath.toAbsolutePath().getParent());
        Files.writeString(resultPath, json, StandardCharsets.UTF_8);
    }

    private static void assertExactConnection(
            Connection connection,
            String expectedDatabase,
            String expectedServerUuid) throws SQLException {
        try (Statement statement = connection.createStatement();
                ResultSet result = statement.executeQuery("SELECT DATABASE(), @@server_uuid")) {
            if (!result.next()
                    || !expectedDatabase.equals(result.getString(1))
                    || !expectedServerUuid.equalsIgnoreCase(result.getString(2))) {
                throw new SQLException("exact_server_identity_guard_failed");
            }
        }
    }

    /** Verifies database and server identity for every connection Flyway opens. */
    private static final class GuardedDataSource implements DataSource {
        private final String jdbcUrl;
        private final String username;
        private final String password;
        private final String expectedDatabase;
        private final String expectedServerUuid;

        GuardedDataSource(
                String jdbcUrl,
                String username,
                String password,
                String expectedDatabase,
                String expectedServerUuid) {
            if (!AUTHORIZED_DATABASE.equals(expectedDatabase)
                    || !AUTHORIZED_DATABASE.equals(databaseFromJdbc(jdbcUrl))) {
                throw new IllegalStateException("exact_database_guard_failed");
            }
            assertCanonicalJdbcUrl(jdbcUrl);
            this.jdbcUrl = jdbcUrl;
            this.username = username;
            this.password = password;
            this.expectedDatabase = expectedDatabase;
            this.expectedServerUuid = expectedServerUuid;
        }

        private Connection open(String connectionUsername, String connectionPassword)
                throws SQLException {
            Connection connection = DriverManager.getConnection(
                    jdbcUrl,
                    connectionUsername,
                    connectionPassword);
            try {
                assertExactConnection(connection, expectedDatabase, expectedServerUuid);
                return connection;
            } catch (Throwable failure) {
                try {
                    connection.close();
                } catch (Throwable ignored) {
                    failure.addSuppressed(ignored);
                }
                if (failure instanceof SQLException) {
                    throw (SQLException) failure;
                }
                throw new SQLException("guarded_connection_identity_failed", failure);
            }
        }

        @Override
        public Connection getConnection() throws SQLException {
            return open(username, password);
        }

        @Override
        public Connection getConnection(String suppliedUsername, String suppliedPassword)
                throws SQLException {
            if (!username.equals(suppliedUsername) || !password.equals(suppliedPassword)) {
                throw new SQLException("guarded_credentials_mismatch");
            }
            return open(suppliedUsername, suppliedPassword);
        }

        @Override
        public PrintWriter getLogWriter() {
            return DriverManager.getLogWriter();
        }

        @Override
        public void setLogWriter(PrintWriter writer) {
            DriverManager.setLogWriter(writer);
        }

        @Override
        public void setLoginTimeout(int seconds) {
            DriverManager.setLoginTimeout(seconds);
        }

        @Override
        public int getLoginTimeout() {
            return DriverManager.getLoginTimeout();
        }

        @Override
        public Logger getParentLogger() {
            return Logger.getLogger("feetfit.flyway.production");
        }

        @Override
        public <T> T unwrap(Class<T> iface) throws SQLException {
            if (iface.isInstance(this)) {
                return iface.cast(this);
            }
            throw new SQLException("not_a_wrapper_for_" + iface.getName());
        }

        @Override
        public boolean isWrapperFor(Class<?> iface) {
            return iface.isInstance(this);
        }
    }

    public static void main(String[] args) throws Exception {
        String stage = "environment";
        Path resultPath = null;
        try {
            String jdbcUrl = requiredEnv("FEETFIT_PRODUCTION_JDBC_URL");
            String username = requiredEnv("FEETFIT_PRODUCTION_DB_USERNAME");
            String password = requiredEnv("FEETFIT_PRODUCTION_DB_PASSWORD");
            String expectedDatabase = requiredEnv("FEETFIT_PRODUCTION_EXPECTED_DATABASE");
            String expectedServerUuid = requiredEnv("FEETFIT_PRODUCTION_EXPECTED_SERVER_UUID");
            resultPath = Path.of(requiredEnv("FEETFIT_PRODUCTION_RESULT_PATH"));

            stage = "target_identity";
            GuardedDataSource guardedDataSource = new GuardedDataSource(
                    jdbcUrl,
                    username,
                    password,
                    expectedDatabase,
                    expectedServerUuid);
            try (Connection ignored = guardedDataSource.getConnection()) {
                // The same guard is applied again to every Flyway connection.
            }

            stage = "configure";
            Flyway flyway = Flyway.configure()
                    .dataSource(guardedDataSource)
                    .locations("classpath:db/migration")
                    .target(MigrationVersion.fromVersion("8"))
                    .cleanDisabled(true)
                    .baselineOnMigrate(false)
                    .ignoreMigrationPatterns("*:pending")
                    .connectRetries(0)
                    .load();

            stage = "repair";
            flyway.repair();

            stage = "validate_before_migrate";
            ValidateResult before = flyway.validateWithResult();
            if (!before.validationSuccessful) {
                throw new IllegalStateException("pre_migrate_validation_failed");
            }
            MigrationInfoService beforeInfo = flyway.info();
            int pendingBefore = beforeInfo.pending().length;
            if (pendingBefore != 6) {
                throw new IllegalStateException("unexpected_pending_before_count");
            }

            stage = "migrate";
            MigrateResult migrateResult = flyway.migrate();
            if (migrateResult.migrationsExecuted != 6) {
                throw new IllegalStateException("unexpected_migrations_executed_count");
            }

            stage = "validate_after_migrate";
            ValidateResult after = flyway.validateWithResult();
            if (!after.validationSuccessful) {
                throw new IllegalStateException("post_migrate_validation_failed");
            }
            MigrationInfoService afterInfo = flyway.info();
            MigrationInfo current = afterInfo.current();
            String currentVersion = current == null || current.getVersion() == null
                    ? ""
                    : current.getVersion().getVersion();
            int pendingAfter = afterInfo.pending().length;
            if (!"8".equals(currentVersion) || pendingAfter != 0) {
                throw new IllegalStateException("target_version_not_reached");
            }

            writeResult(
                    resultPath,
                    "PASS",
                    "complete",
                    migrateResult.migrationsExecuted,
                    pendingBefore,
                    pendingAfter,
                    currentVersion,
                    "");
        } catch (Throwable failure) {
            if (resultPath != null) {
                writeResult(
                        resultPath,
                        "FAIL",
                        stage,
                        0,
                        -1,
                        -1,
                        "",
                        failure.getClass().getName());
            }
            System.err.println(
                    "FEETFIT_PRODUCTION_FLYWAY_FAILED_STAGE=" + stage.toUpperCase(Locale.ROOT));
            System.exit(2);
        }
    }
}
