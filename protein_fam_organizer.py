import os
from os.path import join
from Bio import SeqIO
from Bio.Align.Applications import MuscleCommandline
from Bio.Application import ApplicationError


def create_dict_pantid_uniprotid(file_location, unaligned_fasta_file):
    pro_fam_dict = {}

    with open(file_location) as pro_file:
        for line in pro_file:
            if line != "\n":
                line = line.strip()
                # Split along the "|" mark
                uniprot_id, panther_id = line.split("|")
                if panther_id not in pro_fam_dict:
                    pro_seq = {"id": uniprot_id}
                    pro_fam_dict[panther_id] = [pro_seq]
                else:
                    pro_seq = {"id": uniprot_id}
                    pro_fam_dict[panther_id].append(pro_seq)

    with open(unaligned_fasta_file) as un_fasta:
        sequences = SeqIO.parse(un_fasta, "fasta")

        #Iterate for each sequence
        for seq in sequences:
            for fam, pro_list in pro_fam_dict.items():

                for index, protein_seq in enumerate(pro_list):

                    #Format the ID given into parsable form
                    uniprot_prelim_identif = str(seq.id)
                    _, uniprot_iden, _ = uniprot_prelim_identif.split("|")

                    #Search from the dictionary
                    if uniprot_iden in protein_seq["id"]:
                        up_dict = {"sequence": seq.seq, "header": seq.description}
                        protein_seq.update(up_dict)
                        pro_fam_dict[fam][index] = protein_seq
                        
    return pro_fam_dict

def create_protein_fam_directory(file_location, target_folder):
    pro_fam = set()
    with open(file_location) as protein_fam_files:
        for line in protein_fam_files:
            if line != "\n":
                line = line.strip()
                #Split along the "|" mark
                _, pantherid = line.split("|")
                pro_fam.add(pantherid)

        for fam in pro_fam:
            os.mkdir(join(target_folder, fam))
            print(f"Family {fam} created successfully!")

def store_protein_dictionary(protein_dicti, protein_folder):
    for protein_fam, protein_list in protein_dicti.items():
        pro_fam_file = join(protein_folder, protein_fam, "unalignedproseq.fasta")
        with open(pro_fam_file, "w") as protein_family:
            for pro_seq in protein_list:
                try:
                    formatted_header = pro_seq["header"]+f" {protein_fam}"
                    formatted_seq = format_protein_sequence(str(pro_seq["sequence"]))
                    protein_family.write(">%s\n%s\n" % (formatted_header, formatted_seq))
                except KeyError:
                    print(f"Header of protein ID {pro_seq["id"]} not found")

def format_protein_sequence(sequence, line_length=60):
    formatted_sequence = '\n'.join(sequence[i:i+line_length] for i in range(0, len(sequence), line_length))
    return formatted_sequence

def protein_families_and_sequence_counter(protein_file_path):
    protein_family_count = 0
    protein_sequences_count = 0

    for protein_famiies in os.listdir(protein_file_path):
        #Location of the protein files
        proseq_file_path = os.path.join(protein_file_path, protein_famiies, "unalignedproseq.fasta")

        if os.listdir(os.path.join(protein_file_path, protein_famiies)):
            sequences = SeqIO.parse(proseq_file_path, "fasta")
            for _ in sequences:
                protein_sequences_count += 1
        protein_family_count += 1

    print(f"Number of protein families: {protein_family_count}")
    print(f"Number of sequences: {protein_sequences_count}")

def muscle_align (protein_fam_path):
    # Define file paths
    muscle_exe = "muscle.exe"

    for protein_family in os.listdir(protein_fam_path):
        pro_fam_path = os.path.join(os.path.join(protein_fam_path, protein_family))
        if os.listdir(pro_fam_path):
            input_fasta = os.path.join(pro_fam_path, "unalignedproseq.fasta")

            #Threshold value
            if sum([1 for _ in SeqIO.parse(input_fasta, "fasta")]) < 2:
                print (f"Skipping {protein_family} as the family didn't reach the threshold value!")
                continue

            output_alignment = os.path.join(pro_fam_path, "alignedproseq.fasta")  # Path to save the aligned sequences
            #Create the MUSCLE command
            muscle_cline = MuscleCommandline(muscle_exe, input=input_fasta, out=output_alignment)
            try:
                muscle_cline()
                print("Alignment completed successfully!")
            except ApplicationError:
                print("No sequence on the FASTA file! No alignment to be done!")

def read_all_pro_files_and_write_aligned_seq(protein_fam_parent_path):
    aligned_pro_list = []
    for file_name in os.listdir(protein_fam_parent_path):
        aligned_pro_list.append(os.path.join(protein_fam_parent_path, file_name, "alignedproseq.fasta"))

    for file_path in aligned_pro_list:
        with open("proteinfastaseq/alignedproseq.fasta", "a") as main_align_file:
            try:
                with open(file_path, "r") as aligned_protein_file:
                    for line in aligned_protein_file:
                        main_align_file.write(line)
            except FileNotFoundError:
                print("Aligned file note found in the target folder")

    print("Action Successful!")

def get_max_length(protein_file_path):
    max_len = 0
    seq_count = 0

    sequences = SeqIO.parse(protein_file_path, "fasta")

    for seq in sequences:
        len_ = len(str(seq.seq))
        if len_ > max_len:
            max_len = len_
        seq_count += 1
    print(f"Maximum length: {max_len}, Sequence count: {seq_count}")

if __name__=="__main__":
    #Get parent directory location
    dir_name = os.path.dirname(__file__)

    target_folder_path = join(dir_name, "unaligned_protein_families")
    pro_fam_folder = join(dir_name, "proteinfastaseq", "uniaccwpan.txt")
    target_unaligned_protein_path = join(dir_name, "proteinfastaseq", "unalignedproseq.fasta")
    target_aligned_protein_path = join(dir_name, "proteinfastaseq", "alignedproseq.fasta")

    #Execute
    #create_protein_fam_directory(pro_fam_folder, target_folder_path)

    #Get the protein family dictionary
    #protein_dict = create_dict_pantid_uniprotid(pro_fam_folder, target_unaligned_protein_path)

    #Store to each file as unaligned FASTA
    #store_protein_dictionary(protein_dict, target_folder_path)
    protein_families_and_sequence_counter(target_folder_path)

    #Execute MUSCLE Align to all families
    #muscle_align(target_folder_path)

    #Execute writing to MAIN FILE ALIGNED SEQUENCE
    #read_all_pro_files_and_write_aligned_seq(target_folder_path)

    #Get maximum length for padding
    get_max_length(target_aligned_protein_path)









