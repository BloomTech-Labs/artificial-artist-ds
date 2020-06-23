# Artificial Artist

You can find the project at www.theartificialartist.com.

- [Contributors](#contributors)
- [Project Overview](#project-overview)
    - Tech Stack
    - Data Sources
    - How to connect to the Data Science API
- [Getting Started](#getting-started)
    - Locally
        - Pre-requisites
        - Create Docker Image
- [Contributing](#contributing)
    - Issue/Bug Request
    - Feature Requests
    - Pull Requests
        - Pull Request Guidelines
    - Attribution
- [Documentation](#documentation)


## Contributors
|                                       [Jonathan Mendoza](https://github.com/jonathanmendoza-tx)                                        |                                       [Steve Elliott](https://github.com/StevenMElliott)                                        |
| :-----------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: |
|                      [<img src="https://media-exp1.licdn.com/dms/image/C5603AQFajqe4O-dmIQ/profile-displayphoto-shrink_200_200/0?e=1597881600&v=beta&t=XQOdg_jRPm6OMElW0O4vIwauObyK0WyUh6s3TaQLh2E" width = "200" />](https://github.com/jonathanmendoza-tx)                       |                      [<img src="https://avatars1.githubusercontent.com/u/50522291?s=400&u=a7fbbe3430c3323c4acaf807b5ba093d63718d65&v=4" width = "200" />](https://github.com/StevenMElliott)                       |
|                 [<img src="https://github.com/favicon.ico" width="15"> ](https://github.com/jonathanmendoza-tx)                 |            [<img src="https://github.com/favicon.ico" width="15"> ](https://github.com/StevenMElliott)             |
| [ <img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15"> ](https://www.linkedin.com/in/jonathan-mendoza88/) | [ <img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15"> ](https://www.linkedin.com/in/steven-elliott42/) |


![MIT](https://img.shields.io/packagist/l/doctrine/orm.svg) [![Maintainability](https://api.codeclimate.com/v1/badges/e24ad1860cf0dac6572e/maintainability)](https://codeclimate.com/github/Lambda-School-Labs/artificial-artist-ds/maintainability)

## Project Overview

[Trello Board](https://trello.com/b/48TmCzIE/labs-pt9-artificial-artist)

[Product Canvas](https://www.notion.so/Artificial-Artist-1934140bf39c4f2ba1b8910de0ee0d41)

Artifical Artist is a music visualization platform that use a Generative Adversarial Network to create unqiue videos for our users.

[Deployed Front End] www.theartificialartist.com

### Tech Stack

Python,
AWS Elastic Beanstalk,
Docker,
Gunicorn,
Flask,
Librosa,
and Pytorch

### Data Sources

Deezer API and
Imagenet


### How to connect to the Data Science API

Post request to /entry - which contain atleast a "video_id" and "preview" link from Deezer

## Getting Started

### Locally

#### Pre-requisites 
 - Docker
 - Git and EB CLI
 - .env file with VIS_URL specified as 'http://127.0.0.1:5000/visualize'
 - (optional) .env file with variables: S3_BUCKET, S3_KEY, S3_SECRET_KEY

#### Create Docker Image

 1) Clone repo for artificial-artist-ds
 2) In command line run:

        docker build -t <tag-name> <path of repo>
        docker run -it -d -p 5000:5000 <image-name(tag)>

 3) create a post request and send to http://127.0.0.1:5000/entry
    - Post should include a dictionary with 'params' as first index, and its value will be a dictionary with the keys and values for 'preview' & 'video_id' at a minimum.

    - Example using python:

            import requests

            data = {'params': 
                    {'video_id': 'name_of_video', 
                    'preview': 'url to mp3 file here'
                    }}

            requests.post('http://127.0.0.1:5000/entry', json=data)



## Contributing

When contributing to this repository, please first discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.

Please note we have a [code of conduct](./code_of_conduct.md.md). Please follow it in all your interactions with the project.

### Issue/Bug Request

 **If you are having an issue with the existing project code, please submit a bug report under the following guidelines:**
 - Check first to see if your issue has already been reported.
 - Check to see if the issue has recently been fixed by attempting to reproduce the issue using the latest master branch in the repository.
 - Create a live example of the problem.
 - Submit a detailed bug report including your environment & browser, steps to reproduce the issue, actual and expected outcomes,  where you believe the issue is originating from, and any potential solutions you have considered.

### Feature Requests

We would love to hear from you about new features which would improve this app and further the aims of our project. Please provide as much detail and information as possible to show us why you think your new feature should be implemented.

### Pull Requests

If you have developed a patch, bug fix, or new feature that would improve this app, please submit a pull request. It is best to communicate your ideas with the developers first before investing a great deal of time into a pull request to ensure that it will mesh smoothly with the project.

Remember that this project is licensed under the MIT license, and by submitting a pull request, you agree that your work will be, too.

#### Pull Request Guidelines

- Ensure any install or build dependencies are removed before the end of the layer when doing a build.
- Update the README.md with details of changes to the interface, including new plist variables, exposed ports, useful file locations and container parameters.
- Ensure that your code conforms to our existing code conventions and test coverage.
- Include the relevant issue number, if applicable.
- You may merge the Pull Request in once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you.

### Attribution

These contribution guidelines have been adapted from [this good-Contributing.md-template](https://gist.github.com/PurpleBooth/b24679402957c63ec426).

## Documentation

See [Backend Documentation](https://github.com/Lambda-School-Labs/artificial-artist-be/blob/master/README.md) for details on the backend of our project.

See [Front End Documentation](https://github.com/Lambda-School-Labs/artificial-artist-fe/blob/master/README.md) for details on the front end of our project.

