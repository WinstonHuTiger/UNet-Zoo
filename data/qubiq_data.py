import numpy as np
from data.batch_provider import BatchProvider
from data import qubiq_data_loader


class qubiq_data():

    def __init__(self, sys_config, exp_config):

        data = qubiq_data_loader.load_and_process_data(
            root=exp_config.data_root,
            dataset=exp_config.dataset,
            task=exp_config.task,
            output=exp_config.output,
            preprocessing_folder=sys_config.preproc_folder,
            force_overwrite=True,
        )

        self.data = data

        # Extract the number of training and testing points
        indices = {}
        for tt in data:
            N = data[tt]['images'].shape[0]
            indices[tt] = np.arange(N)

        # Create the batch providers
        augmentation_options = exp_config.augmentation_options

        # Backwards compatibility, TODO remove for final version
        if not hasattr(exp_config, 'annotator_range'):
            exp_config.annotator_range = range(exp_config.num_labels_per_subject)

        self.train = BatchProvider(data['train']['images'], data['train']['labels'], indices['train'],
                                   add_dummy_dimension=False,
                                   do_augmentations=False,
                                   augmentation_options=augmentation_options,
                                   num_labels_per_subject=exp_config.num_labels_per_subject,
                                   annotator_range=exp_config.annotator_range)
        self.validation = BatchProvider(data['val']['images'], data['val']['labels'], indices['val'],
                                        add_dummy_dimension=False,
                                        num_labels_per_subject=exp_config.num_labels_per_subject,
                                        annotator_range=exp_config.annotator_range)
        # self.test = BatchProvider(data['test']['images'], data['test']['labels'], indices['test'],
        #                           add_dummy_dimension=True,
        #                           num_labels_per_subject=exp_config.num_labels_per_subject,
        #                           annotator_range=exp_config.annotator_range)

        # self.test.images = data['test']['images']
        # self.test.labels = data['test']['labels']

        self.validation.images = data['val']['images']
        self.validation.labels = data['val']['labels']


if __name__ == '__main__':

    # If the program is called as main, perform some debugging operations
    from models.experiments import phiseg_rev_7_5_qubiq_4 as exp_config
    from config import local_config
    data = qubiq_data(local_config, exp_config)

    print(data.validation.images.shape)

    print(data.data['val']['images'].shape[0])
    # print(data.data['test']['images'].shape[0])
    print(data.data['train']['images'].shape[0])
    print(data.data['train']['images'].shape[0] + data.data['val']['images'].shape[0])

    print('DEBUGGING OUTPUT')
    print('training')
    for ii in range(2):
        X_tr, Y_tr = data.train.next_batch(2)
        print(np.mean(X_tr))
        print(X_tr.shape)
        print(Y_tr.shape)
        print('--')

    print('validation')
    for ii in range(2):
        X_va, Y_va = data.validation.next_batch(3)
        print(X_va.shape)
        print(Y_va.shape)
        print('--')