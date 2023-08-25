# Script to combine the CPA vallues of the PSR k-means clustering

from pathlib import Path

def main():
    label_file = "label_file_Paht"
    in_dir = "Final_masks_path"
    out_file = "out_file"

    with open(label_file, 'r') as stream:
        labelList = stream.read().splitlines()

    print(labelList)
    combined_list = ["label,SAF	Fibrosis,NAS,F score 1-6,CPA"]
    for cpa_file in Path(in_dir).glob('**/*.txt'):
        i, label = find_match(str(cpa_file), labelList)
        
        if i is not None:
            with open(cpa_file, 'r') as stream:
                cpa_cont = stream.read().splitlines()[0]
            cpa = cpa_cont.split(':')[1]
            label = label.split(',')
            comb_label = ",".join([label[0], label[5], label[6], label[7], cpa])
            combined_list.append(comb_label)
            
            
    with open(out_file, 'w+') as f:
        f.write('\n'.join(combined_list))
    
def find_match(cpa_file_str, labels_loaded):
    for i, label in enumerate(labels_loaded):
        if label.split(',')[0] in cpa_file_str:
            return i, label
    print(cpa_file_str)
    return None, None

main()