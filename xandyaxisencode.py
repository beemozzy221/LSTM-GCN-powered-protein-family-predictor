from Bio import SeqIO
import numpy as np
from prelimyaxisencode import yaxisencode

inputs=255
size_of_seq=1857
aminochar=26
input_file=r"proteinfastaseq/alignedproseq.fasta"
propanidfile=r"proteinfastaseq/uniaccwpan.txt"


def seq2onehot(seq):
    """Create 27-dim embedding"""
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))

    # Convert vocab to one-hot
    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1

    embed_x = [vocab_embed[v] for v in seq]
    seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])

    return seqs_x

def readxyaxis(inputs,size_of_seq,inpufile,uniandpanid,aminochars):
    prelimarr=np.zeros(shape=(size_of_seq),dtype=object)
    xarray=np.zeros(shape=(inputs,size_of_seq,aminochars),dtype=int)
    yarray=np.zeros(shape=(inputs),dtype=object)
    i=0;
    fasta_sequences = SeqIO.parse(open(input_file),'fasta')
    with open(uniandpanid) as propanid:
        for fasta in fasta_sequences:
            name, sequence = fasta.id, str(fasta.seq)
            uniprotindex=name.find("|")
            uniprotindex+=1
            protarr=[]
            while (name[uniprotindex]!="|"):
                protarr.append(name[uniprotindex])
                uniprotindex+=1
            uniprotid="".join(protarr)
  
            for ids in propanid:
                if(ids.rfind(uniprotid)!=-1):
                    index = ids.rindex("|")
                    index+=1
                    arr = []
                    while (ids[index]>" "):
                        arr.append(ids[index])
                        index+=1
                    pantherid = ''.join(arr)
             
                    yarray[i]=pantherid
                            
                    j=0
                    for v in sequence:
                        prelimarr[j]=v
                        j+=1
                    xarray[i]=seq2onehot(prelimarr)
                    i+=1
                    propanid.seek(0)
                    break

    return xarray,yarray




xaxis,prelimyaxis=readxyaxis(inputs,size_of_seq,input_file,propanidfile,aminochar)
yaxisencoder=yaxisencode(prelimyaxis)
yaxis=yaxisencoder.yaxis2onehot()

np.save('auxillaryfiles/protein_encoded_sequence.npy', xaxis)
np.save('auxillaryfiles/protein_family_encoded_labels.npy', yaxis)
                
            
