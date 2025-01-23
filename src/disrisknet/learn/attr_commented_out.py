

      # codes, attr, _, _, _, _ = attribute_batch(lig_code, lig_age, batch, reference_indexes, args)
            # print(codes)
            # print('attributions {}'.format(attr.shape))


# word2attr = defaultdict(list)
    # word2censor_att = defaultdict(partial(defaultdict,list))
    # test_iterator = iter(test_data_loader)
    # for i, batch in enumerate(tqdm(test_iterator)):
    #     batch = train.prepare_batch(batch, args)
    #     attributions_code = lig_code.attribute(inputs=(batch['x']),
    #                                                baselines=None,
    #                                                n_steps=50,
    #                                                return_convergence_delta=False,
    #                                                target=3)
    # print(attributions_code.shape)
    # attributions_code = attributions_code.sum(dim=2).squeeze(0)
    # attributions_code = attributions_code / torch.norm(attributions_code)
    # attributions_code = attributions_code.cpu().detach().numpy()
    # for patient_codes, patient_attr, gold, days in zip(batch['code_str'], attributions_code, batch['y'], batch['days_to_censor']):
    #     patient_codes = patient_codes.split()
    #     time_bin = int(days // 30)
    #     for c, a in zip(patient_codes, patient_attr[-len(patient_codes):]):
    #         code = get_code(args, c)
    #         word2attr[code].append(a)
    #         if gold:
    #             word2censor_att[time_bin][code].append(a)
    # for patient_age, patient_age_attr in zip(ages, add_attr_ages):
    #     word2attr["Add-Age-{}".format(patient_age)].append(patient_age_attr)
    # for patient_age, patient_age_attr in zip(ages, scale_attr_ages):
    #     word2attr["Scale-Age-{}".format(patient_age)].append(patient_age_attr)
    # for patient_age, patient_age_attr in zip(ages, combined_add_ages):
    #     word2attr["Combined-Age-{}".format(patient_age)].append(patient_age_attr)
    # if i >= args.max_batches_per_dev_epoch:
    #     break
    # return word2attr, word2censor_att

    # additional_forward_args=batch)
    # print("lig_code: {}".format(lig_code))
    
    
    #in atttribute batch 
    
            # import pdb; pdb.set_trace()
        # print('explain code...')
        # attributions_code = explain_code.attribute(inputs=(batch['x'],batch['age_seq'], batch['time_seq']),