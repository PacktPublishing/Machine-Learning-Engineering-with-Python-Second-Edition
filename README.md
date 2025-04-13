# Machine-Learning-Engineering-with-Python-Second-Edition

This is the code repository for the [Second Edition of Machine Learning Engineering with Python](https://www.packtpub.com/product/machine-learning-engineering-with-python-second-edition/9781837631964), published by Packt.

More details are below, pick up [your copy](https://a.co/d/8EQvHH2) today!

**Manage the production life cycle of machine learning models using MLOps with practical examples**

The author of this book is - [Andrew Peter McMahon](https://www.linkedin.com/in/andrew-p-mcmahon/)

## About the book
The Second Edition of Machine Learning Engineering with Python is the practical guide that MLOps and ML engineers need to build solutions to real-world problems. It will provide you with the skills you need to stay ahead in this rapidly evolving field.
The book takes an examples-based approach to help you develop your skills and covers the technical concepts, implementation patterns, and development methodologies you need. You'll explore the key steps of the ML development lifecycle and create your own standardized "model factory" for training and retraining models. You'll learn to employ concepts like CI/CD and how to detect different types of drift.
Get hands-on with the latest in deployment architectures and discover methods for scaling up your solutions. This edition goes deeper into all aspects of ML engineering and MLOps, with an emphasis on the latest open-source and cloud-based technologies. This includes a completely revamped approach to advanced pipelining and orchestration techniques.
With a new chapter on deep learning, generative AI, and LLMOps, you will learn to use tools like LangChain, PyTorch, and Hugging Face to leverage LLMs for supercharged analysis. You will explore AI assistants like GitHub Copilot to become more productive, then dive deep into the engineering considerations of working with deep learning.


 
## Key Takeaways
- Plan and manage end-to-end ML development projects
- Explore deep learning, LLMs, and LLMOps to leverage generative AI
- Use Python to package your ML tools and scale up your solutions
- Get to grips with Apache Spark, Kubernetes, and Ray
- Build and run ML pipelines with Apache Airflow, ZenML, and Kubeflow
- Detect drift and build retraining mechanisms into your solutions
- Improve error handling with control flows and vulnerability scanning
- Host and build ML microservices and batch processes running on AWS


## What's New
This second edition goes deeper into ML engineering and MLOps to provide a solid foundation as well as hands-on examples. As well as covering traditional package managers, conda and pip, the second edition shows you how to manage complex dependencies with Poetry. You’ll go beyond creating general ML pipelines with Airflow by building advanced pipelines with ZenML and Kubeflow. The second edition introduces you to Ray, a Python native distributed computing framework, to help ML engineers meet the needs of massive data and massively scalable ML systems. There is a new chapter on **Large Language Models** (**LLMs**), generative AI, and **LLM Operations** (**LLMOps**), using PyTorch, Hugging Face, and GitHub Copilot and consume an LLM via an API using LangChain.


## Outline and Chapter Summary
The book covers a wide variety of **ML Engineering** (**MLE**) and **ML Operations** (**MLOps**) topics to help you understand the tools, techniques, and processes you can apply to engineer your ML solutions, with an emphasis on introducing the key concepts so that you can build on them in your future work. The aim is to develop fundamentals and a broad understanding that will stand the test of time. We cover everything from how to organize your ML team, to software development methodologies and best practices, to automating model building through to packaging your ML code, how to deploy your ML pipelines to a variety of different targets, and then how to scale your workloads for large batch runs. We also discuss, in an entirely new chapter for this second edition, the exciting world of applying ML engineering and MLOps to deep learning and generative AI, including how to start building solutions using LLMs and the new field of LLMOps. Although a far greater emphasis has been placed on using open-source tooling, many examples will also leverage services and solutions from **Amazon Web Services** (**AWS**). 
Machine Learning Engineering with Python, Second Edition will help you to navigate the challenges of taking ML to production and give you the confidence to start applying MLOps in your projects. I hope you enjoy it!


1. [Introduction to ML Engineering](https://github.com/PacktPublishing/Machine-Learning-Engineering-with-Python-Second-Edition/tree/main/Chapter01)
2. [The Machine Learning Development Process](https://github.com/PacktPublishing/Machine-Learning-Engineering-with-Python-Second-Edition/tree/main/Chapter02)
3. [From Model to Model Factory](https://github.com/PacktPublishing/Machine-Learning-Engineering-with-Python-Second-Edition/tree/main/Chapter03)
4. [Packaging Up](https://github.com/PacktPublishing/Machine-Learning-Engineering-with-Python-Second-Edition/tree/main/Chapter04)
5. [Deployment Patterns and Tools](https://github.com/PacktPublishing/Machine-Learning-Engineering-with-Python-Second-Edition/tree/main/Chapter05)
6. [Scaling Up](https://github.com/PacktPublishing/Machine-Learning-Engineering-with-Python-Second-Edition/tree/main/Chapter06)
7. [Deep Learning, Generative AI, and LLMOps](https://github.com/PacktPublishing/Machine-Learning-Engineering-with-Python-Second-Edition/tree/main/Chapter07)
8. [Building an Example ML Microservice](https://github.com/PacktPublishing/Machine-Learning-Engineering-with-Python-Second-Edition/tree/main/Chapter08)
9. [Building an Extract, Transform, Machine Learning Use Case](https://github.com/PacktPublishing/Machine-Learning-Engineering-with-Python-Second-Edition/tree/main/Chapter09)

### Chapter 01, Introduction to ML Engineering
Chapter 1 of "Machine Learning Engineering with Python, Second Edition" provides a comprehensive introduction to the realm of ML engineering and operations. It begins by elucidating the core concepts of ML engineering and MLOps and underscores their importance in the dynamic landscape of ML. The chapter delves into the roles within ML teams and lays out the challenges inherent in ML engineering and MLOps. Moreover, it acknowledges the rapid evolution of ML, with advancements in modeling techniques and technology stacks, necessitating a more profound exploration of various topics to effectively navigate this complex field.

The chapter's primary goal is to equip readers with essential tools and techniques for creating production-ready ML systems in Python. It promises to cover fundamental areas like project management, Python ML package development, and the creation and deployment of reusable training and monitoring pipelines. In addition, it discusses modern tooling, deployment architectures, and scalability using AWS and cloud-agnostic tools. It also introduces readers to transformers and LLMs, exemplified through Hugging Face and OpenAI APIs. Ultimately, the chapter aims to empower readers with a robust foundation for confidently tackling the challenges of ML engineering, regardless of their chosen tools, encouraging further exploration and self-study in pursuit of a successful career in ML engineering.


#### Key Insights:
-	**Dynamic Nature of ML** : The chapter highlights how the field of machine learning has evolved significantly in recent years, with the emergence of more powerful modeling techniques, complex technology stacks, and new frameworks. This evolution underscores the importance of staying up-to-date with the latest trends and tools in ML engineering.
-	**ML Engineering Foundation** : The chapter emphasizes the importance of building a strong foundation in ML engineering. It introduces the core topics that readers will explore in the book, such as project management, Python ML package development, and the creation of training and monitoring pipelines. This foundation is crucial for building production-ready ML systems.
-	**Role of ML Teams** : It provides insights into the various roles within ML teams and explains how these roles complement each other. Understanding these roles is essential for assembling well-resourced teams that can effectively tackle ML projects.
-	**Challenges in Real-World ML** : The chapter discusses the challenges of building ML products within real-world organizations. It underscores the need to estimate value reasonably and communicate effectively with stakeholders, highlighting the practical aspects of ML engineering beyond just technical skills.
-	**High-Level System Design** : Readers are introduced to the concept of high-level ML system design for common business problems. This sets the stage for the technical details to be covered in later chapters and provides a glimpse into what ML solutions should look like from a design perspective.
-	**Motivation for Exploration** : The chapter serves as a motivation for readers to engage with the material in the book and encourages them to embark on a path of exploration and self-study. It emphasizes that a solid conceptual foundation is essential for a successful career as an ML engineer.

### Chapter 02, The Machine Learning Development Process
Chapter 2, "The Machine Learning Development Process," is a comprehensive exploration of organizing and executing successful ML engineering projects. It begins by discussing various development methodologies like Agile, Scrum, and CRISP-DM. The chapter introduces a project methodology developed by the author, which is referred to throughout the book. It also covers essential concepts like **continuous integration/continuous deployment** (**CI/CD**) and developer tools. The chapter's central focus is on delineating how to divide the work for a prosperous ML software engineering project, offering insights into the process, workflow, and necessary tools, accompanied by real ML code examples. It places a particular emphasis on the "four-step" methodology proposed by the author, encompassing the steps of discovery, playing, development, and deployment, comparing it with the popular CRISP-DM methodology in data science.

The chapter provides detailed guidance on setting up tools, version control strategies, CI/CD for ML projects, and potential execution environments for ML solutions. By the chapter's end, readers are well-prepared for Python ML engineering projects, laying the foundation for subsequent chapters. Additionally, it highlights that the concepts discussed here can be applied not only to ML projects but also to other Python software engineering endeavors, emphasizing the versatility of the knowledge presented. Overall, this chapter establishes a robust groundwork for the development and deployment of ML solutions while offering a broader perspective on software engineering practices in Python.

#### Key Insights:
-	**Organizing ML Projects** : The chapter emphasizes the importance of structuring and organizing ML projects effectively. It outlines a four-step methodology (Discover, Play, Develop, Deploy) that serves as a framework for managing ML projects.
-	**Development Methodologies** : It introduces various development methodologies such as Agile, Scrum, and CRISP-DM. The focus is on understanding how these methodologies can be adapted to suit the unique challenges of ML engineering.
-	**Continuous Integration and Deployment** (**CI/CD**): The chapter discusses CI/CD practices, highlighting their significance in ML projects. It provides practical guidance on setting up CI/CD pipelines, including automated model validation, using tools like GitHub Actions.
-	**Tooling and Version Control** : It covers essential tools for ML project development and version control strategies. This includes setting up tools for code development, change tracking, and managing ML projects efficiently.
-	**Project Management Methodologies** : The chapter introduces the CRISP-DM methodology, widely used in data science. It compares this methodology with Agile and Waterfall, offering insights into when and how to apply them in ML projects.
-	**Versatility of Concepts** : It underscores that the principles discussed are not limited to ML projects alone but can be applied to various Python software engineering endeavors. This highlights the broader applicability of the knowledge presented.
-	**Testing and Automation** : The chapter delves into the importance of testing ML code and how to automate testing as part of CI/CD pipelines. It extends the concept to continuous model performance testing and continuous model training.
-	**Project Setup** : Practical steps for setting up tools, environment management, and DevOps and MLOps workflows are detailed. Readers are equipped with the knowledge needed to prepare for Python ML engineering projects.

### Chapter 03, From Model to Model Factory
Chapter 3, titled "From Model to Model Factory," dives into the critical process of standardizing, systematizing, and automating ML model training and deployment. The chapter introduces the concept of the "model factory," a methodology designed for repeatable model creation and validation. It underscores the importance of automating and scaling the challenging task of model training and fine-tuning for production systems. The chapter covers fundamental theoretical concepts necessary for understanding ML models, explores various types of drift detection, and discusses criteria for triggering model retraining.

The chapter begins by emphasizing the central question: How can the intricate process of model training and fine-tuning be automated, reproduced, and scaled for production? It proceeds to provide a comprehensive overview of training different ML models, combining both theoretical and practical insights. The motivation for model retraining, driven by the concept of model performance drift, is elucidated. Additionally, the chapter delves into feature engineering, a crucial aspect of ML tasks, and explains the optimization problems at the heart of ML. It presents various tools for tackling these optimization problems, such as manual model definition, hyperparameter tuning, and **automated ML** (**AutoML**). The chapter also guides readers on interfacing with MLflow APIs for model management and introduces the concept of pipelines to streamline model training steps. Ultimately, the chapter lays the groundwork for understanding how to assemble these components into a cohesive solution, setting the stage for subsequent chapters on packaging and deployment.

#### Key Insights:
-	**The Model Factory Concept** : The chapter introduces the concept of the "model factory," emphasizing the need for a systematic and automated approach to model training and deployment. This approach aims to make the process repeatable and scalable for production systems.
-	**Model Retraining for Drift** : It highlights the importance of model retraining due to the phenomenon of model performance drift. ML models won't perform well indefinitely, and retraining is essential to maintain their effectiveness over time.
-	**Feature Engineering** : The chapter explores the critical role of feature engineering in ML tasks. Feature engineering involves transforming raw data into a format that ML models can understand and learn from effectively.
-	**Understanding Model Training** : Readers gain insights into the technical details of training ML models, including an in-depth look at how ML models learn. The chapter breaks down the complexity of model training into comprehensible components.
-	**Automating Model Training** : It discusses various levels of abstraction for automating the model training process, from manual model definition to hyperparameter tuning and AutoML. Examples of libraries and tools for these tasks are provided.
-	**Drift Detection** : The chapter covers the concept of drift detection, which involves monitoring how ML models and the data they use evolve over time. It includes practical examples of drift detection using packages like Alibi Detect and Evidently.
-	Model Persistence** : It addresses the persistence of ML models, which is crucial for saving and serving models in production systems.
-	**MLflow Model Registry** : The chapter introduces the use of MLflow's Model Registry for programmatically managing the staging of ML models.
-	**Defining Training Pipelines** : It explains how to define training pipelines using libraries like Scikit-Learn and Spark ML. Pipelines are essential for organizing and automating the model training workflow.
-	**Foundation for Deployment** : The chapter establishes a solid foundation for understanding how to package and deploy ML models effectively, setting the stage for subsequent chapters.

### Chapter 04, Packaging Up
Chapter 4, "Packaging Up," delves into the practical aspects of programming in Python, specifically focusing on how to code effectively and package your code for reuse across multiple projects, including those involving ML. It emphasizes that the techniques and methodologies discussed can be applied to various Python development activities throughout an ML project's life cycle. The chapter begins with a recap of fundamental Python programming concepts and moves on to address coding standards, quality code writing, and the distinction between object-oriented and functional programming in Python. It also highlights the importance of testing, logging, error handling, and not reinventing the wheel in the development process. The chapter offers insights into how to package and distribute your code across different platforms and use cases.

This chapter essentially serves as a guide to best practices when creating Python packages for ML solutions. It covers key principles of Python programming, coding standards, and various techniques to ensure high-quality code. The discussion extends to packaging code for distribution, testing, and robust error handling. The chapter culminates by stressing the significance of reusing existing functionality rather than duplicating efforts. As it sets the stage for the next chapter on deployment, it equips readers with the necessary knowledge to prepare their code for deployment on appropriate infrastructure and tools.

#### Key Insights:
-	**Coding Best Practices** : The chapter emphasizes the importance of coding best practices in Python. It highlights the need for writing clean, high-quality code that is both readable and maintainable, regardless of the specific project, including ML tasks.
-	**Coding Standards** : It discusses the significance of adhering to coding standards and guidelines. Consistency in code style and structure, often enforced through tools like linters, contributes to code quality and collaboration.
-	**Object-Oriented vs. Functional Programming** : The chapter explores the differences between object-oriented and functional programming paradigms in Python. It discusses where each approach can be leveraged effectively in ML projects.
-	**Package Development** : It delves into the process of packaging code for reuse across multiple platforms and projects. Readers learn about various tools and setups for packaging code, including Makefiles and Poetry.
-	**Testing and Error Handling** : The chapter underscores the importance of robust testing, logging, and error handling. These are essential components of code that not only function correctly but are also diagnosable when issues arise.
-	**Avoiding Redundancy** : It emphasizes the principle of not reinventing the wheel. Instead of recreating functionality that already exists in the Python ecosystem, developers should leverage existing libraries and packages to streamline their work.
-	**Preparation for Deployment** : The chapter serves as a foundational guide for preparing code for deployment. It lays the groundwork for the next chapter, which will focus on deploying scripts, packages, libraries, and apps on appropriate infrastructure and tools.

### Chapter 05, Deployment Patterns and Tools
Chapter 5, "Deployment Patterns and Tools," delves into the crucial process of deploying ML solutions into real-world production environments. This chapter serves as a bridge between the development of ML models and their practical implementation. It highlights the challenges and importance of transitioning from proof-of-concept to scalable and impactful ML solutions. The chapter begins by focusing on fundamental concepts related to system design and architecture, providing insights into how to build solutions that can be easily scaled and extended. It then introduces the concept of containerization and its role in abstracting application code from specific infrastructure, enhancing portability. A practical example of deploying an ML microservice on AWS is presented. The chapter subsequently returns to the topic of building robust pipelines for end-to-end ML solutions, expanding on the discussion from the previous chapter. It introduces tools like Apache Airflow, ZenML, and Kubeflow for building and orchestrating ML pipelines, offering readers a comprehensive understanding of deployment possibilities.

This chapter equips readers with the knowledge and tools necessary to navigate the complexities of deploying ML solutions effectively. It covers key concepts related to system architecture, containerization, cloud deployment, and pipeline orchestration. By the end of the chapter, readers gain confidence in their ability to deploy and orchestrate complex ML solutions using a variety of software tools. The chapter emphasizes the practical aspects of turning ML models into impactful, production-ready systems that can generate real value.

#### Key Insights:
-	**Deployment Challenges** : The chapter underscores that deploying ML solutions is a challenging but crucial step in the ML development lifecycle. Successful deployment can make the difference between creating value and mere hype.
-	**System Design and Architecture** : It emphasizes the significance of designing and architecting ML systems effectively. Understanding how to develop solutions that can be seamlessly scaled and extended is a fundamental aspect of deployment.
-	**Containerization** : The chapter introduces containerization as a key concept. It explains how containerization abstracts application code from specific infrastructure, enabling portability across various environments.
-	**AWS Microservice Deployment** : A practical example demonstrates how to deploy an ML microservice on AWS. This offers readers insight into the practical aspects of cloud-based ML deployment.
-	**Pipeline Orchestration** : Building on the previous chapter, the discussion on pipeline orchestration continues. Tools like Apache Airflow, ZenML, and Kubeflow are introduced for orchestrating data engineering, ML, and MLOps pipelines.
-	**Scalability**: The chapter sets the stage for scaling ML solutions to handle large volumes of data and high-throughput calculations, hinting at future considerations for ML scalability.
-	**Practical Application** : Throughout the chapter, the practical application of deployment concepts and tools is highlighted, ensuring that readers can apply their knowledge effectively in real-world ML projects.
-	**Value Creation**: Successful deployment is framed as the bridge between ML development and creating real value for customers or colleagues. It emphasizes that the deployment phase is where ML solutions have a tangible impact.

### Chapter 06, Scaling Up
Chapter 6, "Scaling Up," addresses the critical challenge of developing ML solutions that can handle large datasets and high-frequency computations. It recognizes that while running simple ML models on a small scale is suitable for initial exploration and proof of concept, it's inadequate when dealing with massive volumes of data or numerous models. The chapter introduces Apache Spark and Ray frameworks, explaining their inner workings and how to leverage them for scalable ML solutions. It emphasizes the importance of adopting a different approach, mindset, and toolkit for such scenarios. Practical examples illustrate the use of these frameworks for processing large batches of data, and there's an introduction to serverless applications for scaling inference endpoints and containerized ML applications with Kubernetes. The chapter equips readers with the knowledge and tools needed to scale ML solutions to handle larger datasets effectively.

The key takeaways from this chapter are a deeper understanding of Apache Spark and Ray frameworks for distributed computing, including coding patterns and syntax, the significance of **User-Defined Functions** (**UDFs**) for scalable ML workflows, and insights into scaling ML solutions through serverless architectures, containerization, and parallel computing with Ray. The chapter sets the stage for future discussions on scaling ML models, particularly deep learning models and large language models, and emphasizes the importance of these scaling concepts for the rest of the book.

#### Key Insights:
-	**Challenges of Large-Scale Data** : The chapter acknowledges the limitations of running simple machine learning models on small datasets and highlights the challenges of dealing with large volumes of data and high-frequency computations.
-	**Importance of Scalability** : It underscores the importance of adopting a different approach and toolkit when scaling up ML solutions to handle massive datasets or numerous models.
-	**Apache Spark and Rayv** : The chapter provides a detailed exploration of two popular frameworks, Apache Spark, and Ray, for distributed computing. It covers coding patterns, syntax, and the use of UDFs for scalable ML workflows.
-	**Cloud-Based Scaling** : Readers gain insights into using Apache Spark on the cloud, specifically through AWS Elastic MapReduce (EMR), to scale ML solutions.
-	**Serverless Architectures** : The chapter introduces the concept of serverless architectures and demonstrates how to build an ML model serving services using AWS Lambda, emphasizing scalability.
-	**Containerization and Kubernetes** : It discusses horizontal scaling of ML pipelines using containerization and Kubernetes, providing an overview of these tools' benefits and their role in real-time workloads.
-	**Ray Parallel Computing** : Readers learn about the Ray parallel computing framework and its ability to scale compute on heterogeneous clusters for supercharging ML workflows.
-	**Preparation for Future Chapters** : The chapter sets the stage for upcoming discussions on scaling deep learning models, including LLMs, and highlights the importance of the scaling concepts covered in this chapter for the rest of the book.
-	**Prerequisites for Scaling** : It emphasizes that the knowledge and techniques discussed in this chapter are prerequisites for understanding and effectively utilizing the concepts presented in subsequent chapters.

### Chapter 07, Deep Learning, Generative AI, and LLMOps
In Chapter 7, "Deep Learning, Generative AI and LLMOps," the book delves into the rapidly evolving landscape of machine learning and artificial intelligence. It acknowledges the profound changes in the ML and AI realms, with the proliferation of generative artificial intelligence (generative AI or GenAI) tools and large LLMs like ChatGPT, Bing AI, Google Bard, and DALL-E. The chapter aims to guide aspiring ML engineers through this dynamic landscape, providing a comprehensive understanding of the core concepts and foundations necessary to navigate this brave new world effectively.

The chapter commences by revisiting the fundamental algorithmic approach of deep learning, exploring its theoretical underpinnings, and offering insights into building and hosting deep learning models. It then shifts its focus to GenAI and, more specifically, LLMs, such as ChatGPT. It deep dives into the workings and approaches behind these powerful text models, setting the stage for understanding the unique challenges and opportunities they bring. Additionally, the chapter introduces the concept of LLMOps, emphasizing how ML engineering and MLOps principles can be applied to LLMs and highlighting the nascent state of best practices in this domain. Overall, this chapter equips readers with a strong foundation in deep learning, GenAI, LLMs, and the evolving field of LLMOps, positioning them to navigate this transformative landscape confidently.

#### Key Insights:
-	**The Rapid Evolution of AI and ML** : The chapter highlights the rapid pace of development in the AI and ML field. It acknowledges the introduction of generative AI and LLMs like ChatGPT, reflecting the ongoing innovation and growth in AI technology.
-	**Foundational Knowledge** : It emphasizes the importance of building a strong foundation in core AI and ML concepts. Readers are guided through the theoretical aspects of deep learning, ensuring they have a solid understanding of the fundamental principles underpinning these technologies.
-	**Focus on LLMs** : The chapter gives special attention to LLMs, such as ChatGPT, and explores their significance in the AI landscape. It discusses their design principles, behaviors, and practical applications, providing valuable insights for ML engineers.
-	**Introduction to LLMOps** : LLMOps, the application of ML engineering and MLOps to LLMs, is introduced as an emerging field. The chapter outlines its core components and emerging best practices, highlighting that this area is still evolving.
-	**Preparation for Real-World Applications** : By offering a deep understanding of deep learning, GenAI, and LLMs, the chapter prepares ML engineers for real-world applications of these technologies. It equips them to harness the potential of AI and ML in their projects effectively.
-	**Nurturing a Learning Mindset** : The chapter emphasizes that the AI and ML landscape is continually evolving. ML engineers are encouraged to adopt a learning mindset to keep up with the rapid advancements and contribute to the field's growth.

### Chapter 08, Building an Example ML Microservice
Chapter 8, "Building an Example ML Microservice," serves as a practical culmination of the concepts and techniques covered throughout the book. It focuses on creating a machine learning microservice for a forecasting solution, using tools like FastAPI, Docker, and Kubernetes to bring together the knowledge acquired earlier.

The chapter begins by introducing the forecasting problem, emphasizing the importance of understanding the scenario and making key decisions to address it. It then delves into the design of the forecasting service, highlighting the complexities that may arise in real-world ML engineering projects. The selection of appropriate tools is discussed, considering factors like task suitability and developer familiarity. The chapter proceeds to cover training at scale, serving models using FastAPI, and containerizing and deploying the solution to Kubernetes. Each topic provides a practical walk-through, offering readers a valuable reference for tackling similar challenges in their own ML microservices projects. By the end of the chapter, readers gain a clear understanding of how to leverage the tools and techniques acquired throughout the book to build robust ML microservices for diverse business problems, bridging the gap between theory and real-world application.

#### Key Insights:
-	**Realistic Problem Solving** : The chapter demonstrates the importance of understanding the business problem thoroughly before designing a solution. It highlights the need for engineers to make informed decisions that align with the problem's requirements, which often involve dynamically triggered forecasting algorithms.
-	**Tool Selection** : A critical aspect of solving real-world ML problems is selecting the right tools for the job. The chapter discusses the criteria for choosing appropriate tools, considering factors like task suitability and developer familiarity. This insight helps readers make informed choices when building ML solutions.
-	**Microservice Architecture** : The chapter guides readers through the design of a microservice architecture to handle various aspects of ML solutions, such as event handling, model training, storage, and predictions. This architectural approach addresses the complexities of real-world ML engineering projects.
-	**Practical Implementation** : The practical walk-through provided for training models at scale, serving models with FastAPI, and containerizing and deploying the solution to Kubernetes offers readers a concrete example of applying ML engineering concepts and tools to build a robust microservice.

### Chapter 09, Building an Extract, Transform, Machine Learning Use Case
Chapter 9, "Building an Extract, Transform, Machine Learning Use Case," extends the book's lessons by providing a detailed example of a batch-processing ML system. This use case integrates standard ML algorithms with LLMs and LLMOps, demonstrating the application of these concepts in a real-world scenario. The chapter revolves around clustering taxi ride data and performing **natural language processing** (**NLP**) on contextual text data using the **Extract, Transform, Machine Learning** (**ETML**) pattern. It explores the decision-making process, tool selection, and execution of the solution, offering readers practical insights into tackling complex batch ML engineering challenges.

By examining the integration of concepts from previous chapters, such as the ML development process, model packaging, deployment patterns, and orchestration with tools like Apache Airflow, readers gain a holistic understanding of how to approach real-world ML engineering projects. This chapter empowers ML engineers with the knowledge and skills needed to apply ETML patterns, making it an invaluable resource for building robust, batch-processing ML solutions.

#### Key Insights:
-	**ETML Pattern** : The chapter introduces the ETML pattern, which is a common approach in ML solutions. It emphasizes the importance of structuring ML projects to efficiently retrieve, preprocess, and apply machine learning to data in a batch-processing manner. ETML helps in dealing with complex tasks like clustering taxi ride data and performing NLP on textual data.
-	**Integration of Concepts** : Chapter 9 integrates concepts and techniques from earlier chapters, showcasing how ML engineers can leverage the knowledge gained throughout the book. It combines elements from the ML development process, packaging models, deployment patterns, and orchestration using Apache Airflow. This integration highlights the importance of a comprehensive approach to solving real-world ML engineering challenges.
-	**Advanced Tooling** : The chapter delves into the selection of tools and methodologies for building ML solutions effectively. It discusses the use of libraries like Scikit-learn, AWS boto3, and OpenAI APIs for complex functionality. Moreover, it explores advanced features of Apache Airflow, illustrating how to orchestrate ML tasks in a resilient manner.
-	**Real-World Problem Solving** : Through the example scenario of clustering taxi ride data and applying NLP, the chapter demonstrates how to address practical ML engineering challenges. It emphasizes the importance of understanding business requirements and translating them into technical solutions.
-	**Completion of Book** : Chapter 9 marks the conclusion of the book, summarizing the wide range of topics covered in the field of ML engineering. It encourages readers to embrace the exciting opportunities in this evolving field and highlights the growing demand for ML engineering skills.


## Software and hardware list
| Chapter | Software required | Free/Proprietary | Can code testing be performed using a trial? | Cost of the software | Download Links to the software | Hardware specifications | OS required |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Anaconda (>=v22.0) or Miniconda (>=v22.0) | Free | N/A | N/A | [https://www.anaconda.com/download](https://www.anaconda.com/download) | N/A | MacOS, Windows, or Linux |
| 2 | • PyCharm or VSCode <br> • Git and GitHub account <br> • Atlassian JIRA <br> • AWS Account | • Proprietary with free version available <br> • Free <br> • Proprietary with free version available <br> • Proprietary with CLI, etc. free | • N/A <br> • N/A <br> • Yes <br> • Yes | • N/A <br> • N/A <br> • N/A <br> • Free tier, followed by pay as you go | • [https://www.jetbrains.com/pycharm/download](https://www.jetbrains.com/pycharm/download) <br> •[https://code.visualstudio.com/download](https://code.visualstudio.com/download) <br> • [https://www.atlassian.com/git/tutorials/install-git](https://www.atlassian.com/git/tutorials/install-git) <br> • [https://github.com/](https://github.com/) <br> •[https://www.atlassian.com/software/jira](https://www.atlassian.com/software/jira) <br> • [https://aws.amazon.com/](https://aws.amazon.com/) | N/A | MacOS, Windows, or Linux |
| 3 | • MLFlow <br> • Tensorflow <br> • PyTorch | • Free <br> • Free <br> • Free | • N/A <br> • N/A <br> • Yes <br> • Yes | N/A | • [https://mlflow.org/docs/latest/quickstart.html](https://mlflow.org/docs/latest/quickstart.html) <br> •[https://www.tensorflow.org/install](https://www.tensorflow.org/install) <br> • [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) | N/A | MacOS, Windows, or Linux |
| 4 |  Make | Free | N/A | N/A | • [https://formulae.brew.sh/formula/make](https://formulae.brew.sh/formula/make) <br> • Linux: Pre-installed <br> • Recommend you work in Windows Subsystem for Linux and then Make should be available  | N/A | MacOS, Windows, or Linux |
| 5 | • Docker (v20.10) <br> • Kind (>=v0.20) <br> • Kubeflow Pipelines SDK (v1.8) <br> • ZenML (>=0.40) <br> • Apache Airflow (>=v2.6.0)  | Free | N/A | N/A | • [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/) <br> • [https://kind.sigs.k8s.io](https://kind.sigs.k8s.io/) <br> • [https://www.kubeflow.org/docs/started/installing-kubeflow/](https://www.kubeflow.org/docs/started/installing-kubeflow/) <br> • [https://docs.zenml.io/getting-started/installation](https://docs.zenml.io/getting-started/installation) <br> • [https://airflow.apache.org/docs/apache-airflow/stable/installation/](https://airflow.apache.org/docs/apache-airflow/stable/installation/) | N/A | MacOS, Windows, or Linux |
| 6 | • Apache Spark (>=v3.0) <br> • Ray | • Free <br> • Free | N/A | N/A | • [https://spark.apache.org/downloads.html](https://spark.apache.org/downloads.html) <br> • [https://docs.ray.io/en/latest/ray-overview/installation.html](https://docs.ray.io/en/latest/ray-overview/installation.html)  | N/A | MacOS, Windows, or Linux (beta) |
| 7 | • OpenAI API Account | Proprietary | Requires an OpenAI account and API key | Model dependent: $0.0015-0.12/1k tokens | [https://openai.com/blog/openai-api](https://openai.com/blog/openai-api) | N/A | MacOS, Windows, or Linux |
| 8 | • Minikube (v1.30.0) | Free | N/A | N/A | [https://minikube.sigs.k8s.io/docs/start/](https://minikube.sigs.k8s.io/docs/start/)  | N/A | MacOS, Windows, or Linux |
| 9 | • Apache Airflow (>=v2.6.0) | Free | N/A | N/A | [https://airflow.apache.org/docs/apache-airflow/stable/installation/](https://airflow.apache.org/docs/apache-airflow/stable/installation/)  | N/A | MacOS, Windows, or Linux |


## Know more on the Discord server <img alt="Coding" height="25" width="32"  src="https://cliply.co/wp-content/uploads/2021/08/372108630_DISCORD_LOGO_400.gif">
You can get more engaged on the discord server for more latest updates and discussions in the community at [Discord](https://packt.link/mle)


## Download a free PDF <img alt="Coding" height="25" width="40" src="https://emergency.com.au/wp-content/uploads/2021/03/free.gif">

_If you have already purchased a print or Kindle version of this book, you can get a DRM-free PDF version at no cost. Simply click on the link to claim your free PDF._
[Free-Ebook](https://packt.link/free-ebook/9781837631964) <img alt="Coding" height="15" width="35"  src="https://media.tenor.com/ex_HDD_k5P8AAAAi/habbo-habbohotel.gif">


We also provide a PDF file that has color images of the screenshots/diagrams used in this book at [GraphicBundle](https://packt.link/LMqir) <img alt="Coding" height="15" width="35"  src="https://media.tenor.com/ex_HDD_k5P8AAAAi/habbo-habbohotel.gif">


## Get to know the Author
_Andrew Peter McMahon_ has spent years building high-impact ML products across a variety of industries. He is currently Head of MLOps for NatWest Group in the UK and has a PhD in theoretical condensed matter physics from Imperial College London. He is an active blogger, speaker, podcast guest, and leading voice in the MLOps community. He is co-host of the AI Right podcast and was named ‘Rising Star of the Year’ at the 2022 British Data Awards and ‘Data Scientist of the Year’ by the Data Science Foundation in 2019.


## Other Related Books
- [Deep Learning with TensorFlow and Keras – Third Edition](https://www.packtpub.com/product/deep-learning-with-tensorflow-and-keras-third-edition/9781803232911)
- [Mastering Kubernetes - Fourth Edition](https://www.packtpub.com/product/mastering-kubernetes-fourth-edition/9781804611395)



# Errata

## Page 52 - Package management (conda and pip)
In page 52 the text reads :
_"First, if we want to create a conda environment called mleng with Python version 3.8 installed, we simply execute the following in our terminal:_ 
```console
conda env --name mleng python=3.10
```
creating a Conda environment called mleng with Python version 3.8, but the command specifies python=3.10.
To align with the intended version, the correct text should be: 

**First, if we want to create a conda environment called mleng with Python version 3.10 installed**









