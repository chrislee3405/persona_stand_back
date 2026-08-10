# persona_stand_back
a persona chatbot backend

# Prerequisites (one-time setup)
frontend: React/ Vite

backend: Python/ FastAPI

Local Device: Docker Desktop with docker compose installed

GitHub Action Secrets/Variables Setup
    - GitHub repository environment variable: AWS_REGION, AWS_ROLE_ARN, ECR_REPOSITORY

EC2 setup and configuration
    - security group Setting
        Inbound Rules
            SSH (port 22 Source restricted to own IP/VPN only) for Remote server management
            HTTP (port 80 Source can be public or own IP/VPN) for Frontend web application
            Custom TCP (port 8000 Source can be public or own IP/VPN) for FastAPI Backend API
        Outbound Rules
            All Traffic for ECR pulling & general updates
    - Git install, refer to EC2_docker_setup.md

ECR setup and configuration
    - create two private repositories(backend & frontend)
        tag set to immutability while having 'latest' tag as exclusion
    - Push access (GitHub Action): IAM role with AmazonEC2ContainerRegistryPowerUser
    - Pull access (EC2): IAM role with AmazonEC2ContainerRegistryReadOnly

EC2 Server Configuration
    - refer to EC2_docker_setup.md

# Overall work flow (ongoing steps)
Local development on docker environment

push updated version to GitHub repository

Activate GitHub Action
    Actions is scripted by deploy.yml in .github/workflows
    1. Builds Docker images with VITE_API_URL, tagged with both commit SHA and latest
    2. Authenticates & pushes images to AWS ECR
    3. Check the new image appearing in AWS ECR console

docker compose pull & up the image from ECR to EC2 in EC2 terminal
    Owner procedures are list in README.md inside persona_stand_ec2yml folder
    Compose process is scripted in docker-compose.ec2.yml
    If the EC2 instance have stop and restart before, VITE_API_URL of .env inside EC2 must be updated
    Repeat pull & up whenever a new image needs to be deployed
    
run and host the app in EC2 from the images 

# RUN THE BACKEND IN LOCAL DEVICE
.venv\Scripts\activate
uvicorn app.main:app --reload

# RUN THE FRONTEND IN LOCAL DEVICE
npm run dev

# IF ACCIDENTIALLY RUN THE WRONG docker-compose.yml IN EC2 
docker system prune -f

