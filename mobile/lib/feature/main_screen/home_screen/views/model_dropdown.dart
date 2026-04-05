import 'package:ai_dental_studio/feature/select_radio_graph_screen/cubit/select_model_cubit.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:multi_dropdown/multi_dropdown.dart';

import '../../../../core/constants/app_colors.dart';
import '../../../../core/constants/strings.dart';

class ModelDropdown extends StatefulWidget {
  final ValueChanged<List<String>> onChanged;

  const ModelDropdown({super.key, required this.onChanged});

  @override
  State<ModelDropdown> createState() => _ModelDropdownState();
}

class _ModelDropdownState extends State<ModelDropdown> {
  final MultiSelectController<String> _controller =
      MultiSelectController<String>();

  final List<DropdownItem<String>> models = [
    DropdownItem(label: 'YOLOv9', value: 'YOLOv9'),
    DropdownItem(label: 'Faster R-CNN', value: 'Faster R-CNN'),
    DropdownItem(label: 'RetinaNet', value: 'RetinaNet'),
    DropdownItem(label: 'Detectron2', value: 'Detectron2'),
  ];

  @override
  void initState() {
    super.initState();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    final SelectModelCubit cubit = context.read<SelectModelCubit>();

    return BlocBuilder<SelectModelCubit, SelectModelState>(
      builder: (context, state) {
        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            MultiDropdown<String>(
              items: models,
              controller: _controller,
              enabled: true,
              chipDecoration: ChipDecoration(
                backgroundColor: AppColors.black.withAlpha(40),
                wrap: true,
                runSpacing: 2,
                spacing: 8,
                labelStyle: textTheme.bodyMedium?.copyWith(
                  color: AppColors.black,
                ),
              ),
              fieldDecoration: FieldDecoration(
                hintText: 'Default YOLOv9',
                hintStyle: textTheme.bodyLarge?.copyWith(
                  color: AppColors.black.withAlpha(80),
                ),
                showClearIcon: true,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(14),
                  borderSide: BorderSide(color: AppColors.black),
                ),

                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(14),
                  borderSide: BorderSide(color: AppColors.black, width: 2),
                ),
                backgroundColor: AppColors.white,
                suffixIcon: Icon(Icons.arrow_drop_down, color: AppColors.black),
              ),
              dropdownDecoration: DropdownDecoration(
                marginTop: 2,
                maxHeight: 500,
                header: Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Text(
                    Strings.selectYourModels,
                    style: textTheme.titleSmall?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                backgroundColor: AppColors.white,
                borderRadius: BorderRadius.circular(14),
                elevation: 4,
                footer: Container(
                  height: 40,
                  padding: EdgeInsets.only(
                    left: MediaQuery.of(context).size.width / 3,
                    right: MediaQuery.of(context).size.width / 3,
                    top: 6,
                    bottom: 6,
                  ),
                  child: ElevatedButton(
                    onPressed: () {
                      _controller.closeDropdown();
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppColors.black,
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                    child: const Text(Strings.done),
                  ),
                ),
              ),
              dropdownItemDecoration: DropdownItemDecoration(
                selectedIcon: Icon(Icons.check_circle, color: AppColors.green),
                disabledIcon: Icon(Icons.lock, color: Colors.grey),
              ),
              validator: (value) {
                if (value == null || value.isEmpty) {
                  return Strings.pleaseSelectAtLeastOne;
                }
                return null;
              },
              onSelectionChange: (selectedItems) {
                final selectedValues = selectedItems
                    .map(
                      (item) => item.toLowerCase() == 'faster r-cnn'
                          ? 'faster_rcnn'
                          : item,
                    )
                    .toList();
                widget.onChanged(selectedValues);
              },
            ),
          ],
        );
      },
    );
  }
}
