import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

import org.flywaydb.core.Flyway;
import org.flywaydb.core.api.MigrationInfo;
import org.flywaydb.core.api.MigrationInfoService;
import org.flywaydb.core.api.MigrationVersion;
import org.flywaydb.core.api.output.MigrateResult;
import org.flywaydb.core.api.output.ValidateResult;

/** Clone-only Flyway runner. Credentials are accepted only through the process environment. */
public final class FeetfitFlywayCloneRehearsal {
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

    public static void main(String[] args) throws Exception {
        String stage = "environment";
        Path resultPath = null;
        try {
            String jdbcUrl = requiredEnv("FEETFIT_REHEARSAL_JDBC_URL");
            String username = requiredEnv("FEETFIT_REHEARSAL_DB_USERNAME");
            String password = requiredEnv("FEETFIT_REHEARSAL_DB_PASSWORD");
            String cloneDatabase = requiredEnv("FEETFIT_REHEARSAL_CLONE_DATABASE");
            String protectedDatabase = requiredEnv("FEETFIT_REHEARSAL_PROTECTED_DATABASE");
            resultPath = Path.of(requiredEnv("FEETFIT_REHEARSAL_RESULT_PATH"));

            String jdbcDatabase = databaseFromJdbc(jdbcUrl);
            String prefix = "feetfit_flyway_rehearsal_";
            if (!jdbcDatabase.equals(cloneDatabase)
                    || !cloneDatabase.startsWith(prefix)
                    || cloneDatabase.equalsIgnoreCase(protectedDatabase)) {
                throw new IllegalStateException("clone_only_guard_failed");
            }

            stage = "configure";
            Flyway flyway = Flyway.configure()
                    .dataSource(jdbcUrl, username, password)
                    .locations("classpath:db/migration")
                    .target(MigrationVersion.fromVersion("8"))
                    .cleanDisabled(true)
                    .baselineOnMigrate(false)
                    // The required pre-migrate validation runs after repair,
                    // when V3..V8 are legitimately pending.  Applied rows
                    // (V1/V2) are still fully validated; Python additionally
                    // pins their metadata before/after repair.
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

            stage = "migrate";
            MigrateResult migrateResult = flyway.migrate();

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
            System.err.println("FEETFIT_FLYWAY_REHEARSAL_FAILED_STAGE=" + stage.toUpperCase(Locale.ROOT));
            System.exit(2);
        }
    }
}
