# 🐾 Clausecat
Modified Text Classification component which classifies segmented clauses instead of the whole doc.

![Architecture Visualization](https://github.com/thomashacker/healthsea/blob/main/project/scripts/clausecat/img/Clausecat.png)


### Clause Segmentation Component (ToDo)
- Segments docs into clauses and saves the indicies of split in custom attribute `._.clauses`
- Identifies entities and saves the indicies in custom attribute `._.clauses`
- Assigns blinder placeholder in custom attribute `._.clauses`

### Clausecat Component
- Modified textcat component to support custom attribute `._.clauses`
- Uses wrapper model to classify clauses and merges them together into the original doc

### Clausecat Model
- Adds a blinder model before a textcat model
- Blinder uses indices to split doc into clauses and blinds entities

### Clausecat Reader
- Custom corpus reader which transforms textcat annotation into the correct format for clausecat

### Aggregate Health Component (ToDo)
- Uses predictions to aggregate them to entities

## Testing
Use `py.test test` to test:
- Clausecat Reader
- Clausecat Component
