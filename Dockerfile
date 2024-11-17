FROM maven:latest AS builder

WORKDIR /app

COPY . .

RUN mvn clean package

FROM eclipse-temurin:17.0.12_7-jre-jammy

WORKDIR /app

COPY --from=builder /app/target .

CMD ["java", "jar", "my-app-1.0-SNAPSHOT.jar"]