# PPiWD

Backend for PPiWD project.

## Run docker compose

```
docker compose --env-file .env.sample up -d
```

Customize .env.sample file if needed.

## Connect to database from pgAdmin

```
host: timescaledb
port: 5432
user: postgres
password: postgres
```

User and password are defined in .env.sample