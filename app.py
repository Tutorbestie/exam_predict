@app.route('/api/predict', methods=['POST'])
def predict_questions():
    """Generate predicted question paper - works with syllabus OR question papers OR both"""
    try:
        user_id = get_user_id()
        analysis_data = get_analysis_data(user_id)
        
        has_syllabus = analysis_data['syllabus'] is not None
        has_question_papers = len(analysis_data['question_papers']) > 0

        if not has_syllabus and not has_question_papers:
            return jsonify({'error': 'Please upload at least syllabus or question papers and analyze first'}), 400

        if not analysis_data.get('topics'):
            return jsonify({'error': 'Please run analysis first to identify topics'}), 400

        # Get topics - use syllabus topics if available, otherwise use analyzed topics
        syllabus_topics = {}
        if has_syllabus:
            syllabus_topics = analysis_data['syllabus'].get('topics', {})

        question_paper_topics = analysis_data.get('topics', {})
        num_papers = len(analysis_data['question_papers']) if has_question_papers else 1

        # If no topics found, extract from available data
        if not question_paper_topics:
            text_to_analyze = ""
            if has_syllabus:
                text_to_analyze += analysis_data['syllabus'].get('text', '')
            if has_question_papers:
                for paper in analysis_data['question_papers']:
                    text_to_analyze += paper.get('text', '')

            if text_to_analyze:
                keywords = extract_keywords_from_text(text_to_analyze)
                # Only use if we actually found keywords
                if keywords:
                    question_paper_topics = {'General': dict(list(keywords.items())[:20])}
                    analysis_data['topics'] = question_paper_topics
                    update_analysis_data(user_id, analysis_data)

        # Debug info
        logger.info(f"Generating predictions - Syllabus: {has_syllabus}, QP: {has_question_papers}")
        logger.info(f"Syllabus topics: {list(syllabus_topics.keys())}")
        logger.info(f"Question paper topics: {list(question_paper_topics.keys())}")

        predictions = generate_predictions(syllabus_topics, question_paper_topics, max(num_papers, 1))

        # If still no predictions, create some from available topics
        if not predictions and question_paper_topics:
            predictions = []
            for subject, topics in question_paper_topics.items():
                if isinstance(topics, dict):
                    for topic, count in list(topics.items())[:10]:
                        predictions.append({
                            'subject': subject,
                            'topic': topic.title() if topic else 'General Topic',
                            'question': format_question(topic, count),
                            'frequency': count,
                            'probability': min(round(60 + (count * 2), 1), 99),
                            'question_type': 'MCQ (1 mark)',
                            'difficulty': 'Medium'
                        })
                elif isinstance(topics, list):
                    for topic_item in topics[:10]:
                        if isinstance(topic_item, dict):
                            topic_name = topic_item.get('topic', 'Unknown')
                            count = topic_item.get('count', 1)
                            predictions.append({
                                'subject': subject,
                                'topic': topic_name.title(),
                                'question': format_question(topic_name, count),
                                'frequency': count,
                                'probability': min(round(60 + (count * 2), 1), 99),
                                'question_type': 'MCQ (1 mark)',
                                'difficulty': 'Medium'
                            })

            predictions.sort(key=lambda x: x.get('probability', 0), reverse=True)
            predictions = predictions[:20]

        logger.info(f"Generated {len(predictions)} predictions")

        # Save predictions to database
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO predictions (user_id, prediction_results) VALUES (?, ?)',
            (user_id, json.dumps(predictions))
        )
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return jsonify({
            'success': True,
            'predictions': predictions,
            'total_predictions': len(predictions),
            'papers_analyzed': num_papers,
            'message': f'Generated {len(predictions)} predictions'
        }), 200
    except Exception as e:
        logger.error(f"Error in predict_questions: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error during prediction'}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current status of uploaded data"""
    try:
        user_id = get_user_id()
        analysis_data = get_analysis_data(user_id)
        
        return jsonify({
            'syllabus_uploaded': analysis_data['syllabus'] is not None,
            'question_papers_count': len(analysis_data['question_papers']),
            'analysis_completed': 'patterns' in analysis_data and len(analysis_data['patterns']) > 0
        }), 200
    except Exception as e:
        logger.error(f"Error in get_status: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error getting status'}), 500


@app.route('/api/download-prediction', methods=['GET'])
def download_prediction():
    """Download predicted question paper as JSON"""
    try:
        user_id = get_user_id()
        
        # Get latest predictions for this user
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT prediction_results FROM predictions 
            WHERE user_id = ? 
            ORDER BY created_at DESC LIMIT 1
        ''', (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return jsonify({'error': 'No predictions available'}), 400
        
        predictions = json.loads(result[0])
        
        return jsonify({
            'predictions': predictions
        }), 200
    except Exception as e:
        logger.error(f"Error in download_prediction: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error during prediction download'}), 500


@app.route('/api/reset', methods=['POST'])
def reset_data():
    """Reset all uploaded data for current user"""
    try:
        user_id = get_user_id()
        
        # Reset analysis data for this user
        update_analysis_data(user_id, {
            'syllabus': None,
            'question_papers': [],
            'topics': {},
            'patterns': {}
        })
        
        # Clear session data
        session.pop('user_id', None)

        return jsonify({
            'success': True,
            'message': 'All data has been reset'
        }), 200
    except Exception as e:
        logger.error(f"Error in reset_data: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error during reset'}), 500


if __name__ == '__main__':
    # Initialize the database
    init_db()
    # Check if running on Render or similar platform
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Server Error: {error}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    # Return JSON error response
    return jsonify({'error': 'An unexpected error occurred'}), 500
