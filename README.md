# sagemaker-donut-base-finetuned-docvqa



## About this project

This project contains a custom inference file and supporting Jupyter notebook to deploy the donut-base-finetuned-docvqa model hosted on Hugging Face to a SageMaker endpoint. It includes the following files: 

* sagemaker-donut.ipynb - This sample notebook walks you through how to download the Hugging Face model, add the inference.py file, package the model, and deploy the model to a SageMaker endpoint. 

* inference.py - This python file is added to the model artifacts and packaged in a .tar.gz file for deployment to a SageMaker endpoint. This script handles recieving inference in the form of an image and question and handles the transformation logic for the model to create an output. 

If you'd like to run this sample notebook in SageMaker Studio you will need both the proper IAM permission and a S3 bucket to store the model artifact in. 

## Post Deployment Steps

When considering deploying a Hugging Face model to a SageMaker endpoint in a production environment, best practices should be considered. Please refer to the following links for best practices both from Hugging Face and AWS:

* [Use Hugging Face with Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/hugging-face.html)

* [Train and deploy Hugging Face on Amazon SageMaker](https://huggingface.co/docs/sagemaker/en/getting-started)

* [Configure security in Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/security.html)

* [Inference cost optimization best practices](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-cost-optimization.html)

* [Best practices for endpoint security and health with Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/best-practice-endpoint-security.html)