## CS5012 Practical 1: Parts Of SPeech Tagging in the Universal Dependencies Treebanks data 


### Implementation Checklist
- [x] Parameter estimation
- [x] Eager algorithm
- [ ] Viterbi algorithm
- [x] Individually most probable tags
- [x] POS Tagger Evaluation

###### Install the dependencies
```commandline
python3 -m pip install -r requirments.txt
```
##### This POS tagging system implements two main functionalities
* Evaluation
* Testing

### Running System
To use the functionality, we run the following command on the command line.
```
python3 main.py [-h] -r RUN_TYPE [-d DATASET_CODE] [-t TAGGER] [-s TEST_STRING]
```


| Parameter        | Description                                                                                                                                                                                                                                                                             |
|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **RUN_TYPE**     | This selects the functionality to use. This is a required field. Set it to one of these following:<br>`test`: for testing input strings <br/>`eval` For running evaluations and generating metrics and confusion matrices                                                               |
| **DATASET_CODE** | This should point to a directory that contains the treebanks dataset that you are exploring.<br> You set this to the language code provided below:<br><br/>`en`: English dataset<br/>`esp`:Spanish dataset<br/>`du`:Dutch dataset<br/>`all`: Uses all the datasets listed above         |
| **TAGGER**       | This should be set to one of the tagger classes : <br/><br/>`eager`:Uses the Eager algorithm for tagging <br/>`viterbi`: Viterbi algorithm for tagging <br/>`local_decoding`: Local decoding algorithm for individually most probable tags <br/>`all`: Uses all the implemented taggers | 
| **TEST_STRING**  | This is used when the RUN_TYPE is *__test__*. Set it to any string of your choice.<br/> The string would be used to predict tags using the chosen tagger and the chosen dataset                                                                                                         |

<br>

### Examples

To generate tags for the sentence "I am a student in a school" using the eager tagger on the english dataset

```commandline
python3 main.py -r test -d en -t eager -s "I am a student in a school"
```
This should return the following result of the eager tagger for that sentence.
<br/>Each word is attached to its respective part of speech tag in a dictionary as seen below.
```
Generating dataset for en_ewt-ud- Sentences: [Training: 12543], [Test: 2077]
{
    "EagerTagger": {
        "I": "noun",
        "am": "det",
        "a": "aux",
        "student": "noun",
        "in": "det",
        "school": "propn"
    }
}
```


Have Fun :)
