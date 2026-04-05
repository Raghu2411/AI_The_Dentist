import 'package:ai_dental_studio/core/constants/dimens.dart';
import 'package:ai_dental_studio/feature/main_screen/chatbot_screen/chatbot_cubit/chatbot_text_field_cubit.dart';
import 'package:ai_dental_studio/feature/main_screen/chatbot_screen/views/typing_indicator.dart';
import 'package:ai_dental_studio/views/app_progress_indicator.dart';
import 'package:ai_dental_studio/views/empty_view.dart';
import 'package:ai_dental_studio/views/extension/snackbar_extension.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:markdown_widget/widget/markdown.dart';

import '../../../core/constants/app_colors.dart';
import '../../../core/constants/font_sizes.dart';
import '../../../core/constants/strings.dart';
import 'chatbot_cubit/chatbot_cubit.dart';

class ChatbotScreen extends StatefulWidget {
  final String jobId;

  const ChatbotScreen({super.key, required this.jobId});

  @override
  State<ChatbotScreen> createState() => _ChatbotScreenState();
}

class _ChatbotScreenState extends State<ChatbotScreen> {
  late TextTheme textTheme;
  final TextEditingController messageController = TextEditingController();
  final ScrollController scrollController = ScrollController();
  late double screenWidth;

  @override
  void initState() {
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    screenWidth = MediaQuery.of(context).size.width;
    textTheme = Theme.of(context).textTheme;

    return Scaffold(
      backgroundColor: Colors.grey[100],
      appBar: AppBar(
        title: Text(
          Strings.dentalAssistant,
          style: textTheme.headlineSmall?.copyWith(
            color: AppColors.black,
            fontSize: FontSizes.xxMedium,
          ),
        ),
        centerTitle: true,
        backgroundColor: AppColors.white,
      ),
      body: GestureDetector(
        onTap: () {
          FocusScope.of(context).unfocus();
        },
        child: SafeArea(
          child: Column(
            children: [
              Expanded(
                child: BlocConsumer<ChatbotCubit, ChatbotState>(
                  listener: (context, state) {
                    if (state.isLoading) {
                      Future.delayed(const Duration(milliseconds: 100), () {
                        if (scrollController.hasClients) {
                          scrollController.animateTo(
                            scrollController.position.maxScrollExtent,
                            duration: const Duration(milliseconds: 300),
                            curve: Curves.easeOut,
                          );
                        }
                      });
                    }
                    if (state.isError) {
                      context.showSnackBar(
                        context,
                        state.error ?? Strings.someThingWrong,
                        backgroundColor: AppColors.error,
                        textColor: AppColors.white,
                      );
                    }
                  },
                  builder: (context, state) {
                    int historyLength = state.history.length;
                    return widget.jobId == ''
                        ? EmptyView(message: Strings.emptyChatbot)
                        : ListView.builder(
                            controller: scrollController,
                            padding: const EdgeInsets.all(Dimens.large),
                            shrinkWrap: true,
                            reverse: false,
                            itemCount: state.history.length + 1,
                            itemBuilder: (context, index) {
                              return state.isLoading && index == historyLength
                                  ? Padding(
                                      padding: EdgeInsets.only(
                                        top: Dimens.medium,
                                        bottom: Dimens.medium,
                                        right: screenWidth * 0.70,
                                      ),
                                      child: TypingIndicator(),
                                    )
                                  : index < historyLength
                                  ? _buildMessage(index, state, context)
                                  : SizedBox();
                            },
                          );
                  },
                ),
              ),
              if (widget.jobId != '') ...[
                const Divider(height: Dimens.borderXSmall),
                _buildTextField(messageController, context),
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTextField(
    TextEditingController messageController,
    BuildContext context,
  ) {
    ChatbotCubit chatbotCubit = context.read<ChatbotCubit>();
    ChatbotTextFieldCubit cubit = context.read<ChatbotTextFieldCubit>();
    cubit.checkIfFieldEmpty(messageController.text);
    bool isLoading = chatbotCubit.state.isLoading;
    return Padding(
      padding: const EdgeInsets.symmetric(
        horizontal: Dimens.medium,
        vertical: Dimens.marginSmall,
      ),
      child: BlocBuilder<ChatbotTextFieldCubit, ChatbotTextFieldState>(
        builder: (context, state) {
          return Row(
            children: [
              Expanded(
                child: TextField(
                  controller: messageController,
                  decoration: InputDecoration(
                    hintText: Strings.askSomething,
                    filled: true,
                    fillColor: Colors.white,
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(Dimens.radiusXLarge),
                      borderSide: BorderSide.none,
                    ),
                    focusedBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(Dimens.xLarge),
                      borderSide: const BorderSide(
                        color: Colors.black,
                        width: Dimens.borderXSmall,
                      ),
                    ),
                    enabledBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(Dimens.xLarge),
                      borderSide: const BorderSide(
                        color: Colors.grey,
                        width: Dimens.borderXSmall,
                      ),
                    ),
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: Dimens.large,
                    ),
                  ),
                  onChanged: (query) {
                    cubit.checkIfFieldEmpty(query);
                  },
                ),
              ),
              if (state.isEmpty) ...[
                const SizedBox(width: Dimens.marginSmall),
                CircleAvatar(
                  backgroundColor: AppColors.black,
                  child: isLoading
                      ? Padding(
                          padding: const EdgeInsets.all(Dimens.xxMedium),
                          child: AppProgressIndicator(),
                        )
                      : IconButton(
                          icon: const Icon(Icons.send, color: Colors.white),
                          onPressed: () {
                            final text = messageController.text.trim();
                            if (text.isNotEmpty) {
                              context.read<ChatbotCubit>().sendQuery(
                                text,
                                widget.jobId,
                              );
                              messageController.clear();
                            }
                          },
                        ),
                ),
              ],
            ],
          );
        },
      ),
    );
  }

  Widget _buildMessage(int index, ChatbotState state, BuildContext context) {
    final message = state.history[index];
    final isUser = _isUserMessage(message);

    final messageBoxColor = isUser ? AppColors.black : Colors.grey.shade200;

    final textColor = isUser ? Colors.white : Colors.black87;

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.8,
        ),
        margin: const EdgeInsets.symmetric(
          vertical: Dimens.radiusSmall,
          horizontal: Dimens.zero,
        ),
        padding: const EdgeInsets.symmetric(
          horizontal: Dimens.marginMedium,
          vertical: Dimens.medium,
        ),
        decoration: BoxDecoration(
          color: messageBoxColor,
          borderRadius: BorderRadius.only(
            topLeft: Radius.circular(
              isUser ? Dimens.borderMedium : Dimens.zero,
            ),
            topRight: const Radius.circular(Dimens.borderMedium),
            bottomLeft: const Radius.circular(Dimens.borderMedium),
            bottomRight: Radius.circular(
              isUser ? Dimens.zero : Dimens.borderMedium,
            ),
          ),
          border: Border.all(
            color: isUser ? AppColors.primary : AppColors.gray200,
            width: Dimens.borderXSmall,
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withAlpha(1),
              blurRadius: Dimens.radiusSmall,
              offset: const Offset(1, 2),
            ),
          ],
        ),

        child: isUser
            ? Text(
                _messageText(message),
                style: TextStyle(color: textColor, fontSize: FontSizes.xMedium),
              )
            : MarkdownWidget(data: _messageText(message), shrinkWrap: true),
      ),
    );
  }

  bool _isUserMessage(Map<String, String> message) =>
      message['role']?.toLowerCase() == 'user';

  String _messageText(Map<String, String> message) => message['content'] ?? '';
}
