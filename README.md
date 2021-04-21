# Churn Web App

## Background
For the original background, dataset, and model creation for this project, please see my Github repo [Predicting User Churn](https://github.com/joesanfilippo/food_delivery_churn).

Now that I have a model which can predict the probability of a user churning during their first 30 days, I need a way for other teams to interact with it. Whether that is the marketing team targeting users with a high probability or the business intelligence team seeing if the distribution of users is changing over time, having a simple user interface to interact with my model will be crucial.

## Tech Stack
I used several different types of technology to create my web application:

* **Flask**: A web framework and Python module that lets you develop web applications easily. This is the core piece of my web app that will take my model and data stored in a Postgres database and spit out predictions for a user. A majority of what you see is written in HTML that Flask uses to display the web app, but behind the scenes Flask + Python is doing all the work.

* **PostgreSQL**: There are two areas I used PostgreSQL: 
    1. The first is on Food Delivery Company X's database where I can pull udpated data directly from the source. This is where the user information directly lives that my model will need to make predictions. 
    2. The second is a local Postgres database that I created in order to store the user data and my model's predictions. This significantly reduces the time a user has to wait to receive predictions since I am not first querying Company X's database and then running my model on that data. This also simulates a real world scenario where I would create a new table (`churn_predictions`) in a company's database that would update at a regular interval.

* **AWS**: Amazon Web Services (AWS) offers an Elastic Computing (EC2) product that allows anyone to deploy secure, reliable, and scalable websites, apps, or processes. This is what allows me to take anything I do on my local computer and easily replicate it on a website.

* **Docker**: In order to host my web app on AWS, I will need a simple way to create a Postgres database in my EC2 instance. Usually this would involve a host of difficult commands to run once I connect to my EC2 instance, but Docker makes it easy by using the [Postgres image](https://hub.docker.com/_/postgres). 

### Flask

My Flask web-app has 3 main pages and 3 additional "behind-the-scenes" routes that a user can interact with.

* **Home Page**: This page welcomes the user and gives them a basic description about what it's purpose is as well as a prompt to begin.

* **Inputs Page**: This page takes in 3 inputs and uses those inputs to query the Postgres database. 
    1. City IDs: The city id number(s) separated by a `,`. This field is also optional and leaving it blank will return users for all cities.
    2. Lookback Days: The number of days since a user has signed up. The marketing team can use this number to target more recent signups that may not have churned yet but have a high likelihood of churning.
    3. Churn Threshold: This is the minimum probability of churn to filter users by. A number closer than 1 indicates a higher likelihood a user will churn while a number closer to 0 indicates a lower likelihood.

* **Predictions Page**: This page displays the results from the SQL query using the inputs provided. There are five columns I used from my Postgres table: `user_id`, `city_name`, `days_since_signup`, `first_30_day_orders`, and `churn_prediction`.

* **Download Page**: This is a behind-the-scenes page that will take the results from the Predictions Page and download it to the user's computer as a csv file.

* **Retrain Page**: This is a hidden (non-hyperlinked) page that can be used to retrain the model on new data. There is only one button which routes to the Retrained Model page when it is complete.

* **Retrained Model Page**: This page displays the name of the classifier used, the ROC AUC Score from a holdout dataset, and the hyperparameters that were used in the best model.

### PostgreSQL

I wanted to maintain some flexibility on what was ultimately returned to the user from the Predictions page, so while setting up the `churn_predictions` table on my database I included all the features from the original dataset in addition to `city_id` which I will filter on. 

#### `churn_predictions` Schema

| Column Name                    | Data Type   | Description |
|--------------------------------|-------------|-------------|
| user_id                        | Int         | The unique identifier for a user. |
| city_name                      | Varchar(50) | The city the user placed their first order in. |
| city_name                      | Int         | The unique identifier for a city. |
| signup_time_utc                | Date        | The date and time the user signed up on the platform in Coordinated Universal Time. |
| acquisition_channel            | Varchar(50) | The acquisition channel a user signed up through (Facebook, Snapchat, In-restaurant promotions, emails, etc). |
| days_since_signup              | Int         | The number of days since a user has signed up. |
| last_order_time_utc            | Date        | The date and time of the user's last order on the platform in Coordinated Universal Time. |
| signup_to_order_hours          | Double      | The number of hours between signing up and a user's first order. |
| first_order_driver_rating      | Int         | The optional rating a user gave their driver on their first order. Can be -1 if there was no rating. |
| first_order_avg_meal_rating    | Double      | The optional average rating a user gave their meals on their first order. Can be -1 if there were no ratings. An order can have multiple meals included. |
| first_order_meal_reviews       | Int         | The number of meal reviews that a user rated during their first order. An order can have multiple meals included. |
| first_order_delivered_on_time  | Boolean     | Whether or not a user's first order was delivered by the time promised in the app. |
| first_order_hours_late         | Double      | The number of hours late a user's first order was delivered. If an order was delivered on time, this equals 0. |
| first_order_gmv                | Double      | The Gross Merchandise Value (in USD) of a user's first order. This is the total cost of an order, including things like taxes, fees, and tips. |
| first_order_discount_percent   | Double      | The % of the GMV that was discounted on a user's first order. |
| first_order_driver_tips        | Double      | The amount a user tipped a driver on their first order. |
| first_30_day_orders            | Int         | The number of orders a user placed during their first 30 days after their first order. |
| first_30_day_avg_driver_rating | Double      | The average driver rating for all the orders a user placed during their first 30 days after their first order. |
| first_30_day_avg_meal_rating   | Double      | The average meal rating for all the orders a user placed during their first 30 days after their first order. |
| first_30_day_avg_gmv           | Double      | The average GMV (in USD) for all the orders a user placed during their first 30 days after their first order. |
| first_30_day_discount_percent  | Double      | The average discount percent for all the orders a user placed during their first 30 days after their first order. |
| first_30_day_avg_driver_tips   | Double      | The average amount a user tipped a driver for all the orders a user placed during their first 30 days after their first order. |
| first_30_day_subscription_user | Boolean     | Whether or not a user was a subscription user during their first 30 days. |
| first_30_day_support_messages  | Int         | The number of customer support messages a user sent Company X during their first 30 days. Can be -1 if there were no support messages. |
| churn_prediction               | Double      | The probability that a user will churn. |

### AWS

Once I had thoroughly tested my web-application on a local machine, it was time to host it on a remote EC2 instance. This was a fairly simple process but there were several steps along the way to making sure it worked:

1. **Storing private keys**: On a local machine, you can easily reference any keys in your environmet through the `.bash_profile` or `.zshrc file`. When you are running on a remote machine you no longer have that option, so I needed to store those secret keys in AWS's Systems Manager and then call them using Boto3.

2. **Connecting to Postgres Database**: Once I had installed Docker on the EC2 instance, I could start the container. Before that though, I needed to make sure to specify Port: 5432 for TCP/IP connections on the EC2 instance. Once I had done both of those, I had no problem creating a connection to my Postgres database.

3. **Pulling Github Repo**: This is a very simple way to transfer your code from the local machine to a remote machine. Downloading Github and cloning your repo on the EC2 instance will make sure you have all the up-to-date files needed to run.