import os
import tempfile
from unittest.mock import patch

from app import create_app
from models import db, User, Assignment, Submission


def test_submit_evaluation_failure_persists_failed_and_logs(caplog):
    fd, path = tempfile.mkstemp()
    app = create_app('testing')
    app.config.update(SQLALCHEMY_DATABASE_URI=f'sqlite:///{path}', TESTING=True)
    client = app.test_client()
    try:
        with app.app_context():
            db.drop_all()
            db.create_all()
            user = User(student_id='s1', username='s1', usertype='学生', full_name='学生')
            user.password = 'pw'
            db.session.add(user)
            assignment = Assignment(title='旧作业', creator_id='s1')
            db.session.add(assignment)
            db.session.commit()
            assignment_id = assignment.id

        client.post('/login', data={'username': 's1', 'password': 'pw'})
        with patch('routes.api.evaluate_cpp_code', side_effect=RuntimeError('temporary evaluator outage')):
            response = client.post('/api/submit', json={'code': 'int main(){}', 'assignment_id': assignment_id})

        assert response.status_code == 500
        with app.app_context():
            submission = Submission.query.one()
            assert submission.status == 'failed'
            assert 'temporary evaluator outage' in submission.feedback
        assert '代码评估失败' in caplog.text
    finally:
        with app.app_context():
            db.session.remove()
            db.drop_all()
        os.close(fd)
        os.unlink(path)
