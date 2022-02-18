# ArgSciChat: A Dataset of Argumentative Conversational Discussions on Scientific Papers

This is the official repository of the paper "**ArgSciChat: A Dataset of Argumentative Conversational Discussions on Scientific Papers**".

## Cite

Please click [here](https://arxiv.org/abs/2202.06690) for the ArXiv version of the paper.

Please use the following citation (ArXiv):

```
@misc{ruggeri2022argscichat,
      title={ArgSciChat: A Dataset for Argumentative Dialogues on Scientific Papers}, 
      author={Federico Ruggeri and Mohsen Mesgar and Iryna Gurevych},
      year={2022},
      eprint={2202.06690},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Abstract

The applications of conversational agents for scientific disciplines (as expert domains) are understudied due to the lack of dialogue data to train such agents. 
While most data collection frameworks, such as Amazon Mechanical Turk, foster data collection for generic domains by connecting crowd workers and task designers, these frameworks are not much optimized for data collection in expert domains. 
Scientists are rarely present in these frameworks due to their limited time budget. 
Therefore, we introduce a novel framework to collect dialogues between scientists as domain experts on scientific papers.  
Our framework lets scientists present their scientific papers as groundings for dialogues and participate in dialogue they like its paper title. 
We use our framework to collect a novel argumentative dialogue dataset, ArgSciChat. 
It consists of 498 messages collected from 41 dialogues on 20 scientific papers.
Alongside extensive analysis on ArgSciChat, we evaluate a recent conversational agent on our dataset.   
Experimental results show that this agent poorly performs on ArgSciChat, motivating further research on argumentative scientific agents. 
We release our framework and the dataset.

## Project Structure

The project is organized as follows:

* **Data collection tool**: `collection_tool` folder.
* **Allennlp LED baselines**: `argscichat_allennlp` folder.

We provide additional information in each sub-project folder.

In particular, each sub-project folder is meant to be independent of other projects.
Thus, you can work on each sub-project individually.

## Maintainers and Contact

* Internal Maintainer: [Mohsen Mesgar](mailto:mesgar@ukp.informatik.tu-darmstadt.de?subject=[ArgSciChat]%20Repository%20Info)
* External Maintainer: [Federico Ruggeri](mailto:federico.ruggeri6@unibo.it?subject=[ArgSciChat]%20Repository%20Info)

More information at: 
* [ukp.tu-darmstadt.de](https://www.informatik.tu-darmstadt.de/ukp/ukp_home/index.en.jsp) (UKP Lab)
* [tu-darmstadt.de](https://www.tu-darmstadt.de/) (TU Darmstadt)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

## Disclaimer

This repository contains experimental software and is published for the sole purpose of giving additional 
background details on the respective publication.