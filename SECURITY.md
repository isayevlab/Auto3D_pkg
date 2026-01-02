# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 3.0.x   | :white_check_mark: |
| 2.2.x   | :white_check_mark: |
| < 2.2   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in Auto3D, please report it responsibly.

### How to Report

1. **Do NOT open a public issue** for security vulnerabilities
2. Email the maintainers directly at: isayev@andrew.cmu.edu
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Resolution timeline**: Depends on severity

### Severity Levels

| Level | Description | Response Time |
|-------|-------------|---------------|
| Critical | Remote code execution, data loss | 24-48 hours |
| High | Privilege escalation, DoS | 1 week |
| Medium | Information disclosure | 2 weeks |
| Low | Minor issues | Next release |

## Security Best Practices

When using Auto3D:

1. **Keep updated**: Use the latest version
2. **Validate inputs**: Sanitize SMILES strings from untrusted sources
3. **Secure custom models**: Only load models from trusted sources
4. **Environment**: Run in isolated environments for untrusted data

## Known Security Considerations

### Custom Neural Network Models

Loading custom PyTorch models via `optimizing_engine="/path/to/model.pt"` uses `torch.load()`, which can execute arbitrary code. Only load models from trusted sources.

### File Paths

Auto3D reads and writes files based on user-provided paths. Ensure proper access controls in multi-user environments.

## Acknowledgments

We thank security researchers who responsibly disclose vulnerabilities.
