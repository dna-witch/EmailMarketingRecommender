# Reinforcement Learning for Email Marketing Campaign

This project provides code for a microservice that uses Q-learning to predict the best subject line to use to address an email to a customer. The goal is to learn the optimal way to address campaign emails to customers to maximize conversion rate. There are 3 possible choices for the email subject line, and 1 choice to not send an email at all.

### Methods

1. Populate the Q-learning Table: Define states, actions, and rewards based on the available campaign data. Choose appropriate values for the learning rate, discount factor, and exploration-exploitation ratio. 

2. Implementation: Implement Q-learning to build a Q-table that maps out the optimal actions to take depending on the current state of the environment, with the goal of maximizing customer conversion.

3. Performance Evaluation: Evaluate the performance of the Q-learning model by comparing recommendations with the observed results. 

4. Deployment: Deploy the recommendation system via Flask and Docker.

### Endpoints

This container runs a script that listens to POST requests. \
The main script listens to port `8786`.

This microservice returns the following endpoints: \

`/predict`: takes a text input with key "input_data" and returns the recommended action to be taken based on the input. [POST method] \
http://localhost:8786/predict

### Usage Instructions

The Docker image for this project is hosted [here](https://hub.docker.com/repository/docker/shakuntalam/705.603spring24/general) under the tag *email_marketing*.

Clone the Docker image \
`docker pull shakuntalam/705.603spring24:email_marketing`

Find the IMAGE ID using \
`docker image ls`

Run the image using \
`docker run <container ID or container name>`

**Must install and use Postman Client to provide inference parameters and image files.**