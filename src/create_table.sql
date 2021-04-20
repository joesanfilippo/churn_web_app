BEGIN;

CREATE TABLE churn_predictions (
    USER_ID                         INT                 PRIMARY KEY
    ,CITY_NAME                      VARCHAR(50)
    ,CITY_ID                        INT
    ,SIGNUP_TIME_UTC                DATE
    ,ACQUISITION_CHANNEL             VARCHAR(50)
    ,DAYS_SINCE_SIGNUP              INT
    ,LAST_ORDER_TIME_UTC            DATE
    ,SIGNUP_TO_ORDER_HOURS          DOUBLE PRECISION
    ,FIRST_ORDER_DRIVER_RATING      INT
    ,FIRST_ORDER_AVG_MEAL_RATING    DOUBLE PRECISION
    ,FIRST_ORDER_MEAL_REVIEWS       INT
    ,FIRST_ORDER_DELIVERED_ON_TIME  BOOLEAN
    ,FIRST_ORDER_HOURS_LATE         DOUBLE PRECISION
    ,FIRST_ORDER_GMV                DOUBLE PRECISION
    ,FIRST_ORDER_DISCOUNT_PERCENT   DOUBLE PRECISION
    ,FIRST_ORDER_DRIVER_TIPS        DOUBLE PRECISION
    ,FIRST_30_DAY_ORDERS            INT
    ,FIRST_30_DAY_AVG_DRIVER_RATING DOUBLE PRECISION
    ,FIRST_30_DAY_AVG_MEAL_RATING   DOUBLE PRECISION
    ,FIRST_30_DAY_AVG_GMV           DOUBLE PRECISION
    ,FIRST_30_DAY_DISCOUNT_PERCENT  DOUBLE PRECISION
    ,FIRST_30_DAY_AVG_DRIVER_TIPS   DOUBLE PRECISION
    ,FIRST_30_DAY_SUBSCRIPTION_USER BOOLEAN
    ,FIRST_30_DAY_SUPPORT_MESSAGES  INT
    ,CHURN_PREDICTION               DOUBLE PRECISION
);

COMMIT;